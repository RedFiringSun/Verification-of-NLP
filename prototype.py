import torch    # PyTorch for tensor operations and neural network functionalities
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline     # Hugging Face Transformers for NLP models
from sentence_transformers import SentenceTransformer   # For semantic similarity computation
import wikipedia    # For retrieving information from Wikipedia
import json     # For JSON formatting
import re       # For regular expressions
import nltk     # Natural Language Toolkit for text processing
from nltk.tokenize import sent_tokenize     # To split text into sentences
nltk.download('punkt', quiet=True)      # Download tokenizer data quietly
from nltk.corpus import stopwords       # For stop words
nltk.download('stopwords', quiet=True)  # Download stop words data quietly
stop_words = set(stopwords.words('english'))    # Set of stop words like "a", "the", "and", etc.

class NLPVerifier:
    def __init__(self):
        """Initialize the NLP verification system with necessary models and parameters."""
        print("Loading models...")
        
        # Load the paraphrasing model (T5-based)
        self.paraphraser_tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
        self.paraphraser = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
        
        # Load the semantic similarity model (SBERT)
        self.similarity_model = SentenceTransformer("all-mpnet-base-v2")
        
        # Load the question-answering model (Falcon 7B)
        self.qa_pipeline = pipeline(
            "text-generation",
            model="tiiuae/falcon-7b-instruct",
            tokenizer="tiiuae/falcon-7b-instruct",
            torch_dtype=torch.bfloat16,     # Use bfloat16 to reduce memory usage
            device_map={"": "cpu"}          # Use CPU for inference
        )
        print("Models loaded successfully.\n")
        
        # Set thresholds for consistency and fact verification
        self.consistency_threshold = 0.6    # Minimum similarity score for consistent answers
        self.fact_threshold = 0.6           # Minimum similarity score for factual verification
        self.min_consistent_responses = 3   # Minimum number of consistent responses required
        
    def generate_paraphrases(self, question, num_paraphrases=5):
        """Generate different phrasings of the same question."""
        paraphrases = []
        attempts = 0
        max_attempts = num_paraphrases * 3  # Try up to 3x the requested number to ensure uniqueness
        
        # Keep generating until we have enough unique paraphrases
        while len(paraphrases) < num_paraphrases and attempts < max_attempts:
            attempts += 1
            input_text = f"paraphrase: {question}"    # Prefix for T5 paraphraser
            
            # Tokenize the input text
            inputs = self.paraphraser_tokenizer(
                input_text, 
                return_tensors="pt", 
                max_length=128, 
                truncation=True
            )            
            
            # Generate paraphrase using the model
            outputs = self.paraphraser.generate(
                inputs.input_ids,
                max_length=128,
                temperature=0.7,    # Temperature controls randomness (higher = more diverse)
                do_sample=True      # Enable sampling for diverse outputs
            )
            
            # Decode the generated output to text
            paraphrase = self.paraphraser_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Only add unique paraphrases
            if paraphrase not in paraphrases:
                paraphrases.append(paraphrase)
            
        # Return the requested number of paraphrases (or fewer if not enough were generated)
        return paraphrases[:num_paraphrases]
                   
    def get_model_answer(self, question):
        """Get an answer from the QA model for a given question."""
        # Create a prompt for the model
        prompt = f"Answer the question factually and concisely: {question}"
        
        # Generate a response using the QA model
        response = self.qa_pipeline(prompt, max_length=200, do_sample=False)[0]["generated_text"]
        
        # Extract just the answer part by removing the prompt
        answer = response.replace(prompt, "").strip()
        return answer
        
    def check_logical_consistency(self, seed_question):
        """Evaluate if model answers consistently across paraphrased questions."""
        # Get the answer for the original question
        seed_answer = self.get_model_answer(seed_question)
        print(f"Seed answer: {seed_answer}\n")
        
        # Generate paraphrases of the question
        paraphrases = self.generate_paraphrases(seed_question)
        print("Generated paraphrases:")
        for i, paraphrase in enumerate(paraphrases):
            print(f"{i+1}: {paraphrase}")
            
        # Get answers for all paraphrased questions
        paraphrase_answers = []
        for paraphrase in paraphrases:
            paraphrase_answer = self.get_model_answer(paraphrase)
            paraphrase_answers.append(paraphrase_answer)
            
        # Display the answers to paraphrased questions
        print("\nParaphrase Answers:")
        for i, (paraphrase, answer) in enumerate(zip(paraphrases, paraphrase_answers)):
            print(f"{i+1}: {answer}")
            
        # Calculate semantic similarity between original answer and each paraphrased answer
        similarity_scores = []
        for answer in paraphrase_answers:
            # Encode both answers to get vector representations
            embeddings = self.similarity_model.encode([seed_answer, answer])
            
            # Calculate cosine similairity between two embeddings
            similarity = torch.nn.functional.cosine_similarity(
                torch.tensor(embeddings[0]).unsqueeze(0),
                torch.tensor(embeddings[1]).unsqueeze(0),
            ).item()
            similarity_scores.append(similarity)
            
        # Display similarity scores
        print("\nLogical Consistency Similarity Scores:")
        for i, score in enumerate(similarity_scores):
            print(f"Paraphrase {i+1}: {score:.2f}")
            
        # Determine if answers are consistent overall
        consistent_count = sum(1 for score in similarity_scores if score >= self.consistency_threshold)
        is_consistent = consistent_count >= self.min_consistent_responses
        
        # Display consistency result
        print(f"\nLogical Consistency Check (Threshold = {self.consistency_threshold}): {'Passed' if is_consistent else 'Failed'} ({consistent_count}/5 consistent responses)")
            
        # Return all information about the consistency check
        return {
            "seed_question": seed_question,
            "seed_answer": seed_answer,
            "paraphrases": paraphrases,
            "paraphrase_answers": paraphrase_answers,
            "similarity_scores": similarity_scores,
            "is_consistent": is_consistent,
            "consistent_count": consistent_count
        }
        
    def find_best_match(self, claim, content):
        """
        Split the content into chunks (sentences) and compute SBERT similarity.
        Return the best matching chunk, its similarity, and the list of all similarities.
        """
        # A simple split using periods and newlines
        chunks = [chunk.strip() for chunk in re.split(r'\n+|\. ', content) if chunk.strip()]
        if not chunks:
            return {"best_chunk": "", "similarity": 0.0, "all_similarities": []}

        # Encode the claim and chunks
        claim_embedding = self.similarity_model.encode([claim])[0]
        chunk_embeddings = self.similarity_model.encode(chunks)
        
        # Calculate similarities
        similarities = []
        for chunk_embedding in chunk_embeddings:
            similarity = torch.nn.functional.cosine_similarity(
                torch.tensor(claim_embedding).unsqueeze(0),
                torch.tensor(chunk_embedding).unsqueeze(0)
            ).item()
            similarities.append(similarity)
        
        # Find the best match
        if not similarities:
            return {"best_chunk": "", "similarity": 0.0, "all_similarities": []}
        
        top_idx = similarities.index(max(similarities))
        
        return {
            "best_chunk": chunks[top_idx],
            "similarity": similarities[top_idx],
            "all_similarities": similarities
        }
    
    def verify_facts(self, answer):
        """Verify the factual correctness of an answer against Wikipedia."""
        # Split the answer into sentences
        sentences = sent_tokenize(answer)
        if not sentences:
            return {"verfied": False, "reason": "No claims found."}
        
        # Use the first sentence as the main claim to verify
        claim = sentences[0]
        try:            
            # Extract key terms from the claim for better search            
            search_terms = " ".join([word for word in claim.split() if word.lower() not in stop_words])
            
            # Search Wikipedia for relevant articles
            search_results = wikipedia.search(search_terms, results=1)
            
            # If no relevant articles found, return that the claim couldn't be verified
            if not search_results:
                return {
                    "claim": claim,
                    "verfied": False,
                    "similarity": 0,
                    "explanation": f"No relevant Wikipedia content found",
                    "best_match": "",
                    "threshold": self.fact_threshold
                }

            # Get content from the most relevant Wikipedia article
            wiki_title = search_results[0]
            wiki_content = wikipedia.summary(wiki_title)
            
            
            match_result = self.find_best_match(claim, wiki_content)
            
            # Create verification result
            verification_result = {
                "claim": claim,
                "verified": match_result["similarity"] >= self.fact_threshold,
                "similarity": float(match_result["similarity"]),
                "explanation": f"Matches Wikipedia content (similarity: {match_result["similarity"]:.2f})",
                "best_match": match_result["best_chunk"],
                "threshold": self.fact_threshold
            }
            
            # Display verification result
            print("\nFact Verification Result:")
            print(json.dumps(verification_result, indent=2))
            
            return verification_result
                
        except Exception as e:
            # Handle any errors during verification
            error_result = {
                "claim": claim,
                "verified": False,
                "similarity": 0,
                "explanation": f"Error during verification: {str(e)}",
                "best_match": "",
                "threshold": self.fact_threshold
            }
            print("\nFact Verification Result:")
            print(json.dumps(error_result, indent=2))
            return error_result
        
    def evaluate(self, question):
        """Run the full evaluation pipeline on a question."""
        print(f"Seed question: {question}")
        
        # First, check logical consistency
        consistency_result = self.check_logical_consistency(question)
        
        # Then, verify facts in the answer
        fact_result = self.verify_facts(consistency_result["seed_answer"])
        
        # Return combined results 
        return {
            "question": question,
            "consistency_result": consistency_result,
            "fact_result": fact_result
        }

def main():
    """Main function to run the verification system."""
    # Initialize the verifier
    verifier = NLPVerifier()
    
    # List of test questions
    questions = [
        "Who painted the Mona Lisa?",
        "Where is the Eiffel Tower located?",
        "Who developed the Python Programming language?",
        "How many planets are there in the solar system?",
        "What is the capital of Japan?",
        "Who set the first foot on moon?",
        "How many continents are there on Earth?",
        "What is the chemical symbol for gold?",
        "Which planet is known as the Red Planet?",
        "What is the tallest mountain in the world?"
    ]
    
    # Evaluate the first question using questions[0] (number 0 can be changed to test others)
    seed_question = questions[0]
    verifier.evaluate(seed_question)
    
    # To evaluate all questions, uncomment these:
    # for question in questions:
    #     verifier.evaluate(question)

# Entry point of the script
if __name__ == "__main__":
    main()
from collections import defaultdict

def compute_weighted_score(resume_ids,top_k_chunks, top_k_similarity, k, alpha):
    resume_chunk_count = defaultdict(int)  # Number of chunks per resume_id
    resume_similarity_sum = defaultdict(float)  # Sum of similarities per resume_id
    resume_matched_chunks = defaultdict(list)
    resume_final_scores = {}  # Final computed scores per resume_id

    # Process each chunk in the top-k results
    for i in range(len(resume_ids)):
        resume_id = resume_ids[i]
        similarity = top_k_similarity[i] # Convert distance to similarity (higher = better)

        resume_chunk_count[resume_id] += 1  # Count occurrences of this resume_id
        resume_similarity_sum[resume_id] += similarity  # Sum up similarity scores
        resume_matched_chunks[resume_id].append((top_k_chunks[i], similarity))

    # Compute final score for each resume_id
    for resume_id in resume_chunk_count:
        mean_similarity = resume_similarity_sum[resume_id] / resume_chunk_count[resume_id]  # Avg similarity
        chunk_ratio = resume_chunk_count[resume_id] / k  # Fraction of `k` results occupied
        final_score = alpha * mean_similarity + (1 - alpha) * chunk_ratio  # Weighted formula
        resume_final_scores[resume_id] = final_score

    return resume_final_scores, resume_matched_chunks



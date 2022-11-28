import numpy as np

def get_optimal_splits(embeddings, penalty=0.50):
    """
    There is a accumulated score matrix containing the max scores for a segment (i, j),
    containing the sentences [s_i, ..., s_j] at positions i and j. ptr  is pointer for
    backtracking to determine the optimal segment splits while traversing the matrix.
    """
    l0 = embeddings.shape[0]

    dp_mat = np.full((l0, l0), -np.inf, dtype=np.float32)
    colmax = np.full((l0,   ), -np.inf, dtype=np.float32)

    # Backtracking pointer.
    ptr = np.zeros(l0, dtype=np.int32)

    for i in range(l0):
        if i > 0:
            curr_score = colmax[i - 1]
        else:
            curr_score = 0.0

        # Cumulative sum along sentence embedding dimension.
        matcsum = np.cumsum(embeddings[i : i + l0, :], axis=0)

        # Updating the score matrix during forward traversal.
        score = np.linalg.norm(matcsum, axis=1, ord=2)
        dp_mat[i, :matcsum.shape[0]] = curr_score + score - penalty

        # Locating new maxima along the vector embeddings.
        maximas = np.where(dp_mat[i, :matcsum.shape[0]] > colmax[i : i + l0])[0]

        colmax[i + maximas] = dp_mat[i, maximas]
        ptr[i + maximas] = i # Add current pointer.

    path = [ptr[-1]]
    while path[0]: # Insert given a valid path(s).
        path.insert(0, ptr[path[0] - 1])

    return path[1:] # Possible/optimal segmentation.

def get_segmented_sentences(sentences, splits):
    """
    Organize sentences by aggregating them into lists of sentences.
    """
    segmented_sentences = []
    for s, e in zip([0] + splits, splits + [len(sentences)]):
        segmented_sentences.append(sentences[s : e])
        
    return segmented_sentences # List of lists of sentences.
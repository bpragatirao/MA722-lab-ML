def bernoulli_mle(seq):
    seq = list(seq)
    if len(seq) == 0:
        print("Empty sequence.")
    successes = sum(1 for s in seq if s == 1)
    m = len(seq)
    p_hat = successes / m
    return p_hat

# Example
tosses = [1, 0, 1, 0, 0, 1, 0, 1]  # 8 flips
print("MLE p_hat:", bernoulli_mle(tosses))

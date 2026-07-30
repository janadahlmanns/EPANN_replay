"""
Generates a random test genome (population of EPANN CTRNNs) and saves it to disk.

Genome components per individual, all batched over the population dimension:
    W     [pop, N, N]       synaptic weight matrix, sparse
    M     [pop, N, N, N]    modulatory gating tensor, M[b,k,i,j] = neuron k's
                             modulation of synapse (j -> i); sparse, hand-placed
                             on a subset of W's non-zero connections
    A,B,C,D [pop, N, N]     per-synapse plasticity coefficients, dense
    beta  [pop, N]          tonic bias
    eta   [pop, 1]          global plasticity learning-rate scalar

ASSUMPTION (not specified in the brief): all normal draws use mean 0, std 1
(see *_STD constants below). Flag if a different scale is wanted.
"""

# ==== 1. IMPORTS ============================================================
import torch

# ==== 2. CONSTANTS / USER INPUTS ===========================================
POP = 5                        # population size
N = 20                         # neurons per network
DEVICE = "cuda"

GENOME_SEED = 42               # RNG seed for genome generation only

W_CONNECTIVITY_FRAC = 0.10     # fraction of all N*N possible connections that are non-zero
W_NONZERO_COUNT = int(W_CONNECTIVITY_FRAC * N * N)   # 40 for N=20
W_POS_COUNT = W_NONZERO_COUNT // 2                   # 20
W_NEG_COUNT = W_NONZERO_COUNT - W_POS_COUNT          # 20
W_STD = 1.0                    # std of the normal distribution weight magnitudes are drawn from

MOD_TARGET_COUNT = 4           # number of W connections that receive a modulator
MOD_POS_COUNT = 2
MOD_NEG_COUNT = 2
MOD_STD = 1.0                  # std of the normal distribution modulation magnitudes are drawn from

ABCD_STD = 1.0                 # std for A, B, C, D plasticity coefficients
BETA_STD = 1.0                 # std for tonic bias
ETA_STD = 1.0                  # std for global plasticity scalar

OUTPUT_PATH = "data/test_genome.pt"


# ==== 3. GENOME GENERATION FUNCTION ========================================
def generate_random_genome(
    pop, n,
    w_nonzero_count, w_pos_count, w_neg_count, w_std,
    mod_target_count, mod_pos_count, mod_neg_count, mod_std,
    abcd_std, beta_std, eta_std,
    device, generator,
):
    """Build a random population genome. Sparsity pattern for W (and the M
    targets drawn from it) is independent per individual."""

    W = torch.zeros(pop, n, n, device=device)
    M = torch.zeros(pop, n, n, n, device=device)

    # --- 3.1 sparse weight matrix W, fixed pos/neg split per individual ---
    w_signs_template = torch.cat([
        torch.ones(w_pos_count, device=device),
        -torch.ones(w_neg_count, device=device),
    ])

    mod_sign_template = torch.cat([
        torch.ones(mod_pos_count, device=device),
        -torch.ones(mod_neg_count, device=device),
    ])

    for b in range(pop):
        # pick w_nonzero_count distinct (i, j) connections out of n*n, self-connections allowed
        flat_idx = torch.randperm(n * n, generator=generator, device=device)[:w_nonzero_count]
        rows, cols = flat_idx // n, flat_idx % n

        magnitudes = torch.randn(w_nonzero_count, generator=generator, device=device).abs() * w_std
        signs = w_signs_template[torch.randperm(w_nonzero_count, generator=generator, device=device)]
        W[b, rows, cols] = magnitudes * signs

        # --- 3.2 modulation: pick mod_target_count of the just-placed connections ---
        target_pick = torch.randperm(w_nonzero_count, generator=generator, device=device)[:mod_target_count]
        target_rows, target_cols = rows[target_pick], cols[target_pick]

        # each of the mod_target_count targets gets its own independent random source neuron
        source_neurons = torch.randint(0, n, (mod_target_count,), generator=generator, device=device)

        mod_magnitudes = torch.randn(mod_target_count, generator=generator, device=device).abs() * mod_std
        mod_signs = mod_sign_template[torch.randperm(mod_target_count, generator=generator, device=device)]
        M[b, source_neurons, target_rows, target_cols] = mod_magnitudes * mod_signs

    # --- 3.3 dense plasticity coefficients, tonic bias, global eta ---
    A = torch.randn(pop, n, n, generator=generator, device=device) * abcd_std
    B = torch.randn(pop, n, n, generator=generator, device=device) * abcd_std
    C = torch.randn(pop, n, n, generator=generator, device=device) * abcd_std
    D = torch.randn(pop, n, n, generator=generator, device=device) * abcd_std
    beta = torch.randn(pop, n, generator=generator, device=device) * beta_std
    eta = torch.randn(pop, 1, generator=generator, device=device) * eta_std

    return {"W": W, "M": M, "A": A, "B": B, "C": C, "D": D, "beta": beta, "eta": eta}


# ==== 4. MAIN EXECUTION =====================================================
if __name__ == "__main__":
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(GENOME_SEED)

    genome = generate_random_genome(
        POP, N,
        W_NONZERO_COUNT, W_POS_COUNT, W_NEG_COUNT, W_STD,
        MOD_TARGET_COUNT, MOD_POS_COUNT, MOD_NEG_COUNT, MOD_STD,
        ABCD_STD, BETA_STD, ETA_STD,
        DEVICE, gen,
    )

    torch.save(genome, OUTPUT_PATH)

    print(f"Saved genome to {OUTPUT_PATH}")
    for key, tensor in genome.items():
        nnz = torch.count_nonzero(tensor).item()
        print(f"  {key:5s} shape={tuple(tensor.shape)}  nonzero={nnz}")

"""
Generates a random test genome (population of EPANN CTRNNs) and saves it to disk.

Genome components per individual, all batched over the population dimension. Matches
the pivot in sim_core/genome_codec.py: the genome holds ONLY the plasticity rule +
tonic bias -- no initial weights (those are sampled fresh every lifetime, see
sim_core.genome_codec.sample_initial_weights, not reproduced here since this script
only exercises the genome side for plotting):
    M       [pop, N, N, N]    modulatory gating tensor, M[b,k,i,j] = neuron k's
                               modulation of synapse (j -> i); sparse, hand-placed
                               on a random subset of (i,j) connections
    A,B,C,D [pop, N, N]       per-synapse plasticity coefficients, dense
    beta    [pop, N]          tonic bias
    eta     [pop, N, N]       per-synapse plasticity learning rate, dense

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

MOD_CONNECTION_COUNT = 40      # number of (i, j) connections eligible to receive a modulator target
MOD_TARGET_COUNT = 4           # number of those connections that actually receive a modulator
MOD_POS_COUNT = 2
MOD_NEG_COUNT = 2
MOD_STD = 1.0                  # std of the normal distribution modulation magnitudes are drawn from

ABCD_STD = 1.0                 # std for A, B, C, D plasticity coefficients
BETA_STD = 1.0                 # std for tonic bias
ETA_STD = 1.0                  # std for per-synapse plasticity learning rate

OUTPUT_PATH = "data/test_genome.pt"


# ==== 3. GENOME GENERATION FUNCTION ========================================
def generate_random_genome(
    pop, n,
    mod_connection_count, mod_target_count, mod_pos_count, mod_neg_count, mod_std,
    abcd_std, beta_std, eta_std,
    device, generator,
):
    """Build a random population genome. M's sparsity pattern (which (k,i,j) triples
    are non-zero) is independent per individual."""

    M = torch.zeros(pop, n, n, n, device=device)

    mod_sign_template = torch.cat([
        torch.ones(mod_pos_count, device=device),
        -torch.ones(mod_neg_count, device=device),
    ])

    # --- 3.1 modulation: pick mod_target_count connections out of a random candidate pool ---
    for b in range(pop):
        flat_idx = torch.randperm(n * n, generator=generator, device=device)[:mod_connection_count]
        rows, cols = flat_idx // n, flat_idx % n

        target_pick = torch.randperm(mod_connection_count, generator=generator, device=device)[:mod_target_count]
        target_rows, target_cols = rows[target_pick], cols[target_pick]

        # each of the mod_target_count targets gets its own independent random source neuron
        source_neurons = torch.randint(0, n, (mod_target_count,), generator=generator, device=device)

        mod_magnitudes = torch.randn(mod_target_count, generator=generator, device=device).abs() * mod_std
        mod_signs = mod_sign_template[torch.randperm(mod_target_count, generator=generator, device=device)]
        M[b, source_neurons, target_rows, target_cols] = mod_magnitudes * mod_signs

    # --- 3.2 dense plasticity coefficients, tonic bias, per-synapse eta ---
    A = torch.randn(pop, n, n, generator=generator, device=device) * abcd_std
    B = torch.randn(pop, n, n, generator=generator, device=device) * abcd_std
    C = torch.randn(pop, n, n, generator=generator, device=device) * abcd_std
    D = torch.randn(pop, n, n, generator=generator, device=device) * abcd_std
    beta = torch.randn(pop, n, generator=generator, device=device) * beta_std
    eta = torch.randn(pop, n, n, generator=generator, device=device) * eta_std

    return {"M": M, "A": A, "B": B, "C": C, "D": D, "beta": beta, "eta": eta}


# ==== 4. MAIN EXECUTION =====================================================
if __name__ == "__main__":
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(GENOME_SEED)

    genome = generate_random_genome(
        POP, N,
        MOD_CONNECTION_COUNT, MOD_TARGET_COUNT, MOD_POS_COUNT, MOD_NEG_COUNT, MOD_STD,
        ABCD_STD, BETA_STD, ETA_STD,
        DEVICE, gen,
    )

    torch.save(genome, OUTPUT_PATH)

    print(f"Saved genome to {OUTPUT_PATH}")
    for key, tensor in genome.items():
        nnz = torch.count_nonzero(tensor).item()
        print(f"  {key:5s} shape={tuple(tensor.shape)}  nonzero={nnz}")

from pyphi.examples import basic_noisy_selfloop_network
tpm = basic_noisy_selfloop_network().tpm
state = (0, 0, 1)  # A network state is a binary tuple

print(tpm)

print(tpm[(1,1,1)])
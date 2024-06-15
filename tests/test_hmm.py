# import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd


def dptable(state_prob):
    print(" ".join(("%8d" % i) for i in range(state_prob.shape[0])))
    for i, prob in enumerate(state_prob.T):
        print("%.7s: " % states[i] + " ".join("%.7s" % ("%f" % p) for p in prob))


# Just to highlight the maximum in the Series (color : Yellow)
def highlight_max(s):
    is_max = s == s.max()
    return ["background-color: yellow" if v else "" for v in is_max]


from HiddenMarkovModel import HiddenMarkovModel

p0 = np.array([0.6, 0.4])

emi = np.array([[0.5, 0.1], [0.4, 0.3], [0.1, 0.6]])

trans = np.array([[0.7, 0.3], [0.4, 0.6]])

states = {0: "Healthy", 1: "Fever"}
obs = {0: "normal", 1: "cold", 2: "dizzy"}
obs_seq = np.array([0, 0, 1, 2, 2])

df_p0 = pd.DataFrame(p0, index=["Healthy", "Fever"], columns=["Prob"])
df_emi = pd.DataFrame(
    emi, index=["Normal", "Cold", "Dizzy"], columns=["Healthy", "Fever"]
)
df_trans = pd.DataFrame(
    trans, index=["fromHealthy", "fromFever"], columns=["toHealthy", "toFever"]
)

# ### Inital state probability

df_p0

# ### Transition Probability Matrix

df_trans

# ### The Emission Probability Matrix

df_emi

# ### Now, time to Run Viterbi

model = HiddenMarkovModel(trans, emi, p0)
states_seq, state_prob = model.run_viterbi(obs_seq, summary=False)

print("Observation sequence: ", [obs[o] for o in obs_seq])
df = pd.DataFrame(state_prob.T, index=["Healthy", "Fever"])
df.style.apply(highlight_max, axis=0)

print("Most likely States: ", [states[s] for s in states_seq])

# # ### Define Model Parameters

# p0 = np.array([0.5, 0.5])

# emi = np.array([[0.9, 0.2], [0.1, 0.8]])

# trans = np.array([[0.7, 0.3], [0.3, 0.7]])

# states = {0: "rain", 1: "no_rain"}
# obs = {0: "umbrella", 1: "no_umbrella"}

# obs_seq = np.array([1, 1, 0, 0, 0, 1])

# # ## Run Forward-Backward

# from forward_backward_model import HiddenMarkovModel_FB

# model = HiddenMarkovModel_FB(trans, emi, p0)

# results = model.run_forward_backward(obs_seq)
# result_list = ["Forward", "Backward", "Posterior"]

# for state_prob, path in zip(results, result_list):
#     inferred_states = np.argmax(state_prob, axis=1)
#     print()
#     print(path)
#     dptable(state_prob)
#     print()

# print("=" * 60)
# print("Most likely Final State: ", states[inferred_states[-1]])
# print("=" * 60)

# # ## Run Baum-Welch

# # ### Data Generator Function


# def generate_HMM_observation(num_obs, pi, T, E):
#     def drawFrom(probs):
#         return np.where(np.random.multinomial(1, probs) == 1)[0][0]

#     obs = np.zeros(num_obs)
#     states = np.zeros(num_obs)
#     states[0] = drawFrom(pi)
#     obs[0] = drawFrom(E[:, int(states[0])])
#     for t in range(1, num_obs):
#         states[t] = drawFrom(T[int(states[t - 1]), :])
#         obs[t] = drawFrom(E[:, int(states[t])])
#     return obs, states


# # ### True Parameters that Generated the data

# True_pi = np.array([0.5, 0.5])

# True_T = np.array([[0.85, 0.15], [0.12, 0.88]])

# True_E = np.array([[0.8, 0.0], [0.1, 0.0], [0.1, 1.0]])

# # ### Generate a Sample of 50 Observations

# obs_seq, states = generate_HMM_observation(50, True_pi, True_T, True_E)

# print("First 10 Obersvations:  ", obs_seq[:18])
# print("First 10 Hidden States: ", states[:18])

# # ### Initialize to Arbitrary Parameters

# init_pi = np.array([0.5, 0.5])

# init_T = np.array([[0.5, 0.5], [0.5, 0.5]])

# init_E = np.array([[0.3, 0.2], [0.3, 0.5], [0.4, 0.3]])

# # ### Train Model

# model = HiddenMarkovModel(init_T, init_E, init_pi, epsilon=0.0001, maxStep=12)

# trans0, transition, emission, c = model.run_Baum_Welch_EM(
#     obs_seq, summary=False, monitor_state_1=True
# )

# print("Transition Matrix: ")
# print(transition)
# print()
# print("Emission Matrix: ")
# print(emission)
# print()
# print("Reached Convergence: ")
# print(c)

# # ### Plot of Probability of State 1 over multiple training steps

# plt.figure(figsize=(15, 6))
# plt.plot(1 - model.state_summary[[0, 4, 6, 8, 9, 10]].T)
# plt.ylim(-0.1, 1.1)
# plt.title("Probability State=1 over time")
# plt.xlabel("Time")
# plt.draw()

# # ### Plot of True State over Guess Probability of State=1

# plt.figure(figsize=(15, 6))
# plt.plot(states.T, "-o", alpha=0.7)
# plt.plot(1 - model.state_summary[-2].T, "-o", alpha=0.7)
# plt.legend(("True State", "Guessed Probability of State=1"), loc="right")
# plt.ylim(-0.1, 1.1)
# plt.xlabel("Time")
# plt.draw()

# # ### Beware of Overfitting
# #
# # * The algorithm is clearly learning to generate the correct hidden state.
# # * Baum Welch does, however, overfit quickly.
# # * It is important to:
# #     * Train on multiple sequences.
# #     * Regularize training.
# #     * Repeat inference with multiple random initial parameters.

# pred = (1 - model.state_summary[-2]) > 0.5
# print("Accuracy: ", np.mean(pred == states))

# # ## Visualize the Computation Graph with Tensorboard
# #
# # Each nodes expends into an exploded view of the computation graph generated by Tensorflow.
# # Check it out!
# #
# # **Run Tensorboard on command line :**
# #
# # > tensorboard --logdir= [graph/file/location](https://github.com/MarvinBertin/HiddenMarkovModel_TensorFlow/tree/master/TensorBoard/Baum-Welch)
# #

# # ![](images/BW1.png)

# # # That's it for now...Feel free to add more

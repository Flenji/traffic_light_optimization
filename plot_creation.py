
import utility

agent_suffix = "_reward4_randtraining"
scores = utility.load_object("scores"+agent_suffix, "results")
epsilons = utility.load_object("epsilons"+agent_suffix, "results")

#utility.plot_learning_curve(scores["B1"], epsilons, filename = "model"+agent_suffix, path="results", mean_over=360)
utility.plot_learning_curve(scores, epsilons, filename = "model_"+agent_suffix, path="results", mean_over=2400)
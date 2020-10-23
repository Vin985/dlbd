import os


from dlbd.evaluation.evaluator import Evaluator


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# stream = open("src/evaluation_config.yaml", "r")
# opts = yaml.load(stream, Loader=yaml.Loader)

# print(opts)

# for model in opts["models"]:
#     versions = model.get("versions", 1)
#     print(type(versions))

evaluator = Evaluator(opts_path="src/evaluation_config.yaml")

res = evaluator.evaluate()

# res.to_csv("results/evaluation/stats3.csv")
print(res)

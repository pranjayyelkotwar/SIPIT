import torch


stats = torch.load(
      "experiment_outputs/exp1_100/feature_statistics.pt",
      map_location="cpu",
      weights_only=True,
  )

hle_means = stats["hle_mean"]
hle_prevalence = stats["hle_prevalence"]

arc_means = stats["arc_mean"]
arc_prevalence = stats["arc_prevalence"]

hle_dict = {}
arc_dict = {}

with open("experiment_outputs/exp1_100/hle_feature_stats.txt", "w") as output:
    for feature_id, (mean, prevalence) in enumerate(
        zip(hle_means.tolist(), hle_prevalence.tolist())
    ):
        if prevalence >= 0.3:
            hle_dict[feature_id] = {'mean' : mean, 'prevalence' : prevalence}
            print(
                f"{feature_id:>12} "
                f"{mean:>18.8f} "
                f"{prevalence:>11.2%}",
                file=output,
                flush=True,
            )


with open("experiment_outputs/exp1_100/arc_feature_stats.txt", "w") as output:
    for feature_id, (mean, prevalence) in enumerate(
        zip(arc_means.tolist(), arc_prevalence.tolist())
    ):
        if prevalence <= 0.05:
            arc_dict[feature_id] = {'mean' : mean, 'prevalence' : prevalence}
            print(
                f"{feature_id:>12} "
                f"{mean:>18.8f} "
                f"{prevalence:>11.2%}",
                file=output,
                flush=True,
            )


hle_keys = hle_dict.keys()
arc_keys = arc_dict.keys()

intersection = list(hle_keys & arc_keys)
print(len(intersection))
intersection.sort()
print(intersection)
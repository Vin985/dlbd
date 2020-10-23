#%%
import feather
import pandas as pd

df_norel = feather.read_dataframe(
    "/mnt/win/UMoncton/Doctorat/data/dl_training/datasets/mel_32_2048_1024/citynet/training_tags.feather"
)

df_rel = feather.read_dataframe(
    "/mnt/win/UMoncton/Doctorat/data/dl_training/datasets/mel_32_2048_1024/arctic/training_tags.feather"
)


if "related" not in df_norel.columns:
    df_norel["related"] = ""

df_rel[df_rel.related.isnull()] = ""

df_norel2 = df_norel.copy()
df_rel2 = df_rel.copy()
classes_df = pd.read_csv("src/classes.csv")
class_list = classes_df.loc[classes_df.class_type == "biotic"].tag.values

print(class_list)

#%%
%%timeit

def f1(x, classes):
    if x.related:
        tags = ",".join([x.tag, x.related]).lower().split(",")
    else:
        tags = [x.tag]
    for tag in tags:
        if tag in classes:
            return True
    return False


res = df_norel.apply(f1, axis=1, classes=class_list)


#%%
def f2(df, classes):
    has_related = "related" in df.columns
    cols = ["tag"]
    if has_related:
        cols.append("related")
    df2 = df[cols]
    res = [False] * df.shape[0]
    i = 0
    for row in df2.itertuples(name=None, index=False):
        tags = ",".join(row).lower().split(",")
        for tag in tags:
            if tag in classes:
                res[i] = True
        i += 1
    return res


#%%
%%timeit
res = df_norel[f2(df_norel, class_list)]
print(res)

#%%
%%timeit
res = df_rel[f2(df_rel, class_list)]


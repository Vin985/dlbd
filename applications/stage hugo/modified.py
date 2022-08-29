#%%
from pathlib import Path
import os
import pandas as pd

dir_root = Path(
    "/mnt/win/UMoncton/OneDrive - Université de Moncton/Data/Reference/Arctic/Complete"
)
dest_dir = Path("/mnt/win/UMoncton/Doctorat/dev/dlbd/applications/stage hugo/results")
global_path = dest_dir / "global_labels.csv"
grouped_path = dest_dir / "grouped_labels.csv"
ref_classes_path = Path(
    "/mnt/win/UMoncton/Doctorat/dev/dlbd/examples/resources/reference_classes.csv"
)
corrected_path = dest_dir / "corrected_labels.csv"

#%%
# Alire: https://peps.python.org/pep-0008/

# Ajout d'argument a la fonction pour eviter d'utiliser une valeur definie en dehors et eviter
# les effets de bord
def assemblage(src_dir, final_dest_path, gb_dest_path, ext=".csv"):
    res = []
    # Tu utilises pathlib pour generer ton chemin, autant ne pas melanger les styles
    if not final_dest_path.exists():
        file_list = list(src_dir.rglob("*" + ext))
        for file_path in file_list:
            path_plot = file_path.parts[-2]
            path_year = file_path.parts[-3]
            try:
                print("Loading file " + str(file_path))
                df = pd.read_csv(file_path)
                df["year"] = path_year
                df["plot"] = path_plot
                res.append(df)
            except Exception:
                print("Error, impossible to read file {}".format(file_path))

        ############################## Zone de sauvegarde + concaténation ##############################
        df_final = pd.concat(res).drop_duplicates().reset_index(drop=True)
        df_final.to_csv(final_dest_path, index=False)
    else:
        df_final = pd.read_csv(final_dest_path)
    df_gb = df_final.groupby(["Label", "Related"]).agg(
        "count"
    )  # Enregistrement d'un csv brute de Label et de Related

    df_gb.reset_index().to_csv(gb_dest_path, index=False)

    return df_final, df_gb


final, grouped = assemblage(dir_root, global_path, grouped_path)

#%%

###########################################################
##########################             ##########################
#############################   Feature   #############################
##########################             ##########################
###########################################################

#######################################################################


def replace_related(df, df_ref):
    # Fonction de d'harmonisation des Relateds en fonction des Labels
    for (Label, Related) in df_ref.itertuples(index=False):
        df.loc[
            df.Label == Label, "Related"
        ] = Related  # Modification du Related en fonction du Label de reference
    df.loc[df.Related.isnull(), "Related"] = ""
    return df


ref_classes = pd.read_csv(ref_classes_path)
corrected = replace_related(final, ref_classes)
corrected.to_csv(corrected_path, index=False)


corrected.groupby(["Label", "Related"]).count()

#%%

##################################################################
#########################   Feature Dispatch   #########################
##################################################################


def Dispatch_Plot_Year():
    df = pd.read_csv(name4)
    try:
        del df["Unnamed: 0"]
    except:
        pass
    for (Plot, Year), group in df.groupby(["Plot", "Year"]):
        Year = str(Year)
        try:
            os.makedirs(name5 / Essai0 / Year)
        except FileExistsError:
            pass
        group.to_csv(f"{name5/Essai0/Year/Plot}.csv", index=False)


def Dispatch_Label():
    df = pd.read_csv(name4)
    for (Label), group in df.groupby(["Label"]):
        try:
            os.makedirs(name5 / Essai)
        except FileExistsError:
            pass
        group1 = group.groupby(["Label"]).sum()
        del group1["id"]
        del group1["Year"]
        del group1["noise"]
        group1.to_csv(f"{name5/Essai/Label}.csv", index=False)


def Unpack():
    Dispatch_Plot_Year()
    Dispatch_Label()
    fenetre = Tk()
    label = Label(fenetre, text="Dispatch done")
    label.pack()
    b1 = Button(fenetre, text="Ok", command=fenetre.destroy)
    b1.pack()
    fenetre.mainloop()


###############################################################
#########################   Feature Stats   #########################
###############################################################


def Stat_Mean():
    df_Stat_Mean = pd.read_csv(name4)  # Ouverture du fichier rangé
    try:
        del df_Stat_Mean["Unnamed: 0"]
    except:
        pass
    df_Stat = (
        df_Stat_Mean.loc[
            :, ["MaxAmp", "MinAmp", "MeanAmp", "MinimumFreq_Hz", "MaximumFreq_Hz"]
        ]
        .groupby(df_Stat_Mean["Label"])
        .mean()
    )
    df_Stat.columns = [
        "MaxAmp_Mean",
        "MinAmp_Mean",
        "MeanAmp_Mean",
        "MinimumFreq_Hz_Mean",
        "MaximumFreq_Hz_Mean",
    ]
    df_Stat.to_csv(f"{name5/Essai1}.csv", index=True)  # Enregistrement du fichier
    return df_Stat


###########################################################################


def Stat_Max():
    df_Stat_Max = pd.read_csv(name4)  # Ouverture du fichier rangé
    try:
        del df_Stat_Max["Unnamed: 0"]
    except:
        pass
    df_Stat = (
        df_Stat_Max.loc[
            :, ["MaxAmp", "MinAmp", "MeanAmp", "MinimumFreq_Hz", "MaximumFreq_Hz"]
        ]
        .groupby(df_Stat_Max["Label"])
        .max()
    )
    df_Stat.columns = [
        "MaxAmp_Max",
        "MinAmp_Max",
        "MeanAmp_Max",
        "MinimumFreq_Hz_Max",
        "MaximumFreq_Hz_Max",
    ]
    df_Stat.to_csv(f"{name5/Essai2}.csv", index=False)  # Enregistrement du fichier
    return df_Stat


###########################################################################


def Stat_Min():
    df_Stat_Min = pd.read_csv(name4)  # Ouverture du fichier rangé
    try:
        del df_Stat_Min["Unnamed: 0"]
    except:
        pass
    df_Stat = (
        df_Stat_Min.loc[
            :, ["MaxAmp", "MinAmp", "MeanAmp", "MinimumFreq_Hz", "MaximumFreq_Hz"]
        ]
        .groupby(df_Stat_Min["Label"])
        .min()
    )
    df_Stat.columns = [
        "MaxAmp_Min",
        "MinAmp_Min",
        "MeanAmp_Min",
        "MinimumFreq_Hz_Min",
        "MaximumFreq_Hz_Min",
    ]
    df_Stat.to_csv(f"{name5/Essai3}.csv", index=True)  # Enregistrement du fichier
    return df_Stat


###########################################################################


def Stat_Std():
    df_Stat_Std = pd.read_csv(name4)  # Ouverture du fichier rangé
    try:
        del df_Stat_Std["Unnamed: 0"]
    except:
        pass
    df_Stat_Std["Sing_Time_Seconds"] = (
        df_Stat_Std["LabelEndTime_Seconds"] - df_Stat_Std["LabelStartTime_Seconds"]
    )  # Creation d'une colonne qui recup le temps de chaque chant
    df_Stat = (
        df_Stat_Std.loc[:, ["Sing_Time_Seconds"]].groupby(df_Stat_Std["Label"]).std()
    )
    df_Stat.columns = ["Std"]
    df_Stat.to_csv(f"{name5/Essai5}.csv", index=True)  # Enregistrement du fichier
    return df_Stat


###########################################################################


def Stat_Global():
    df_Stat_Global = pd.read_csv(name4)  # Ouverture du fichier rangé
    try:
        del df_Stat_Global["Unnamed: 0"]
    except:
        pass
    df_Label = pd.read_csv(name3, delimiter=";")
    try:
        del df_Label["Unnamed: 0"]
    except:
        pass
    df_Stat = (
        df_Stat_Global.loc[:, ["LabelStartTime_Seconds", "LabelEndTime_Seconds"]]
        .groupby(df_Stat_Global["Label"])
        .sum()
    )
    df2 = df_Stat_Global["Label"].value_counts()
    frames = [df_Stat, df2]  # Assemblage
    A, B, C, D = Stat_Min(), Stat_Max(), Stat_Mean(), Stat_Std()
    result = (
        pd.concat(frames, axis=1).drop_duplicates().reset_index(drop=True)
    )  # Assemblage concaténation
    result["Total_Sing"] = (
        result["LabelEndTime_Seconds"] - result["LabelStartTime_Seconds"]
    )  # Durée total de chant
    result["Mean_Sing"] = result["Total_Sing"] / result["Label"]  # Temps moyen de chant
    frames1 = [A, B, C, D]
    result1 = pd.concat(frames1, axis=1).drop_duplicates().reset_index(drop=True)
    frames2 = [result1, result]
    result3 = pd.concat(frames2, axis=1).drop_duplicates().reset_index(drop=True)
    result3["Labels"] = df_Label["Label"]
    result3.set_index("Labels", inplace=True)
    result3.to_csv(
        f"{name5/Essai4}.csv", index=True
    )  # Enregistrement du fichier    #Enregistrement du fichier

    fenetre = Tk()
    label = Label(fenetre, text="Done")
    label.pack()
    b1 = Button(fenetre, text="Ok", command=fenetre.destroy)
    b1.pack()
    b2 = Button(fenetre, text="Graph", command=Graph_Global)
    b2.pack()
    fenetre.mainloop()

    return result3


###########################################################################

###############################################################
#########################   Feature Manip   #########################
###############################################################


def Stat_Label():
    try:
        global subset
        df = pd.read_csv(name4)
        try:
            del df["Unnamed: 0"]
        except:
            pass
        df2 = df["Label"].value_counts()
        df3 = (
            df.loc[:, ["LabelStartTime_Seconds", "LabelEndTime_Seconds"]]
            .groupby(df["Label"])
            .sum()
        )
        df3["Total_Sing"] = df3["LabelEndTime_Seconds"] - df3["LabelStartTime_Seconds"]
        del df3["LabelEndTime_Seconds"], df3["LabelStartTime_Seconds"]
        df3["Total_Tag"] = df2
        df3["Mean_Sing_Seconds"] = df3["Total_Sing"] / df3["Total_Tag"]
        df4 = df.loc[:, ["background"]].groupby(df["Label"]).sum()
        df3["Background"] = df4
        df3["%_Background"] = df3["Background"] * 100 / df3["Total_Tag"]
        Noice1 = df[df["noise"] == 1].groupby(df["Label"]).sum()
        Noice1["%Noice"] = Noice1["noise"] * 100 / df3["Total_Tag"]
        Noice2 = (df[df["noise"] == 2].groupby(df["Label"]).sum()) / 2
        Noice2["%Noice"] = Noice2["noise"] * 100 / df3["Total_Tag"]
        Noice3 = (df[df["noise"] == 3].groupby(df["Label"]).sum()) / 3
        Noice3["%Noice"] = Noice3["noise"] * 100 / df3["Total_Tag"]
        (A, B, C) = (
            round(Noice1["%Noice"], 2),
            round(Noice2["%Noice"], 2),
            round(Noice3["%Noice"], 2),
        )
        (
            df3["Rain"],
            df3["Rain%"],
            df3["Wind"],
            df3["Wind%"],
            df3["Both"],
            df3["Both3%"],
        ) = (Noice1["noise"], A, Noice2["noise"], B, Noice3["noise"], C)
        df3["Index"] = df3.index.drop_duplicates()
        df3.to_csv(f'{name5/"Monki"}.csv', index=True)
        df3_dic = df3.loc[subset].to_dict()
        print("Bonjour")
        fenetre = Tk()
        label = Label(fenetre, text=df3.loc[subset])
        label.pack()
        b1 = Button(fenetre, text="Ok", command=fenetre.destroy)
        b1.pack()
        b2 = Button(fenetre, text="Graph", command=Graph_Label)
        b2.pack()
        fenetre.mainloop()
        return (df3.loc[subset], df3_dic)
    except:
        fenetre = Tk()
        label = Label(
            fenetre,
            text="Unknown keyword, check the writing and be careful with the case",
        )
        label.pack()
        label1 = Label(fenetre, text="Back to main menu")
        label1.pack()
        b1 = Button(fenetre, text="Ok", command=fenetre.destroy)
        b1.pack()
        fenetre.mainloop()


###########################################################################


def Input():
    def Jsp():
        root = Tk()
        root.geometry("300x200")

        def action(event):
            # Obtenir l'élément sélectionné
            global subset
            subset = listeCombo.get()
            print("Vous avez sélectionné : '", subset, "'")

        labelChoix = Label(root, text="Veuillez faire un choix !")
        labelChoix.pack()

        # 2) - créer la liste Python contenant les éléments de la liste Combobox
        df1 = pd.read_csv(name3, delimiter=";")
        test = df1["Label"].tolist()
        listeLabel = test

        # 3) - Création de la Combobox via la méthode ttk.Combobox()
        listeCombo = ttk.Combobox(root, values=listeLabel)
        listeCombo.current(0)

        listeCombo.pack()
        listeCombo.bind("<<ComboboxSelected>>", action)
        b3 = Button(root, text="Next", font=("times", 18), command=Stat_Label)
        b3.pack()
        b4 = Button(root, text="Leave", font=("times", 18), command=root.destroy)
        b4.pack()

        root.mainloop()

    fen = Tk()
    b1 = Button(fen, text="Per Label", font=("times", 18), command=Jsp)
    b2 = Button(fen, text="Global", font=("times", 18), command=Stat_Global)
    b3 = Button(fen, text="Leave", font=("times", 18), command=fen.destroy)
    b1.pack(side="top")
    b2.pack()
    b3.pack(side=("bottom"))
    fen.mainloop()


###########################################################################


def Graph_Label():
    print("test")


###########################################################################


def Graph_Global():
    print("test")


###############################################################
#########################    Main Window    #########################
###############################################################

lafen = Tk()

b1 = Button(lafen, text="Assemblage", font=("times", 18), command=Assemblage)
b2 = Button(lafen, text="Input", font=("times", 18), command=Input)
b3 = Button(lafen, text="Sort", command=Sort)
b4 = Button(lafen, text="Dispatch", command=Unpack)
b5 = Button(lafen, text="Leave", command=lafen.destroy)

b1.pack()
b2.pack()
b3.pack()
b4.pack()
b5.pack()

lafen.mainloop()

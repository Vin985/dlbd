import pathlib
import os
import pandas as pd
import tkinter as Tk
from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfilename, askdirectory

####################################  Partie Ajout de colonnes + concaténation ####################################

# root = Tk.Tk()                                                          #Court-circuite la fenetre graphique
# root.withdraw()                                                         #Lance la fênetre de selection fichier Global pour reussir l'ajout de la concaténation
# name1 = askopenfilename()                                              #Selection du fichier de sauvegarde
# root.mainloop()

Essai = "Ordered_Files"
Essai0 = "Ordered_Files_"
Essai1 = "Stat_Label_Mean"
Essai2 = "Stat_Label_Max"
Essai3 = "Stat_Label_Min"
Essai4 = "Stat_Label_Global"
Essai5 = "Stat_Label_Std"

name1 = pathlib.Path(r"C:\Users\zepht\Desktop\Global\global.csv")
name2 = pathlib.Path(r"C:\Users\zepht\Desktop\Global\global_gb.csv")
name3 = pathlib.Path(r"C:\Users\zepht\Desktop\Global\sort.csv")
name4 = pathlib.Path(r"C:\Users\zepht\Desktop\Global\global_sort.csv")
name5 = pathlib.Path(r"C:/Users/zepht/Desktop/Global")
name6 = pathlib.Path(r"C:\Users\zepht\Desktop\Global\Ordered_Files.csv")

dir_root = pathlib.Path(
    r"C:\Users\zepht\Desktop\Test_dossier"
)  # askdirectory()      #Selection du dossier dans lequel appliquer le traitement

df_sort = pd.read_csv(name3, sep=";")  # Ouverture du fichier de reference des Labels
df = pd.read_csv(name4)

try:
    del df["Unnamed: 0"]
    del df["Unnamed: 0.1"]
    del df["Unnamed: 0.1.1"]
    del df["Unnamed: 0.1.1.1"]
except:
    pass


def Assemblage():
    res = []
    # Sylvain: n<utilise pas de variables definies en dehors de ta fonction.
    # Utilise des arguments a la place
    for root, dirs, files in os.walk(
        dir_root
    ):  # Automatisation par loop. Il faut trouver un moyen de pouvoir selection le dossier sur lequel on va appliquer le script
        for file in files:  # Pour chaque fichier dans le dossier:
            if file.endswith(".csv"):  # Si le fichier est une extension.csv alors:
                try:
                    files = os.path.join(root, file)
                    # print(files)                                           #Balise
                    path_windows = pathlib.Path(files)  # On recupere son chemin d'accès
                    PathPlot = path_windows.parts[
                        -2
                    ]  # On  and convertit le plot et l'année du dossier contenant le fichier en variable
                    PathYear = path_windows.parts[-3]
                    # Ouverture du fichier à modifier + ajout de la colonne year et plot
                    df = pd.read_csv(files)
                    try:
                        del df["Unnamed: 0"]
                    except:
                        pass
                    try:
                        df.loc[df.Label == "Dog?", "Label"] = "Dog"
                    except:
                        pass
                    df["Year"] = PathYear
                    df["Plot"] = PathPlot
                    res.append(df)
                    print(df)  # balise
                except Exception:
                    print("Error, impossible to read file {}".format(files))

    ############################## Zone de sauvegarde + concaténation ##############################
    df_final = (
        pd.concat(res).drop_duplicates().reset_index(drop=True)
    )  # Concaténation sans doublon
    df_gb = df_final.groupby(
        ["Label", "Related"]
    ).sum()  # Enregistrement d'un csv brute de Label et de Related

    fenetre = Tk()
    label = Label(fenetre, text="Done")
    label.pack()
    b1 = Button(fenetre, text="Ok", command=fenetre.destroy)
    b1.pack()
    fenetre.mainloop()  # Balise

    df_final.to_csv(name1)
    df_gb.to_csv(name2)

    ###########################################################


##########################             ##########################
#############################   Feature   #############################
##########################             ##########################
###########################################################

#######################################################################


def Sort():
    df = pd.read_csv(name1)
    try:
        df.loc[df.Label == "Dog?", "Label"] = "Dog"
        df.to_csv(name1)
    except:
        pass

    try:
        df.loc[df.Label == "Unknow sound", "Label"] = "UNKN"
        df.to_csv(name1)
    except:
        pass

    try:
        df.loc[df.Label == "Sandpiper (chip)", "Label"] = "Sandpiper (chirp)"
        df.to_csv(name1)
    except:
        pass
    try:
        del df["Unnamed: 0"]
    except:
        pass
    df_sort = pd.read_csv(name3, delimiter=";")
    try:
        del df_sort["Unnamed: 0"]
    except:
        pass
    # Fonction de d'harmonisation des Relateds en fonction des Labels
    for (Label, Related) in df_sort.itertuples(index=False):
        df.loc[
            df.Label == Label, "Related"
        ] = Related  # Modification du Related en fonction du Label de reference
    df_final_sort = df
    df_final_sort.to_csv(name4)
    fenetre = Tk()
    b1 = Button(fenetre, text="Done", command=fenetre.destroy)
    b1.pack()
    fenetre.mainloop()
    return df


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

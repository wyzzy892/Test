import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, shutil

class PlotData:
    def r2_score(self, y_true:np.ndarray=None, y_pred:np.ndarray=None)->np.double:
        """Calculate r2 score metric"""
        return 1. - np.sum((y_true - y_pred))/np.sum((y_true-y_true.mean()))
    
    def accuracy(self, y_true:np.ndarray=None, y_pred:np.ndarray=None)->np.double:
        """Calculate accuracy"""
        return np.sum(y_true == y_pred)/len(y_true)

    def draw_plots(self, file:str=None)->None:
        """Plot meaningful statistics"""
        # checking the correctness of the input
        df = None
        try:
            if file is None:
                raise Exception("No file was provided!")
            elif not file.endswith('.json'):
                raise Exception("Wrong format of the file was provided!")
            else:
                df = pd.read_json(file)
        except Exception as e:
            print(e)
            return
        
        # remove directory for plots if  exists and create new one
        if os.path.exists("./plots"):
            shutil.rmtree("./plots")
        os.mkdir("./plots")

        # plot comparative histograms (mean.png, max.png, min.png)
        cols = ["mean", "max", "min"]
        for col in cols:
            plt.figure(figsize=(10, 6))
            plt.hist(df[col], label=col, color="cyan", edgecolor="black")
            plt.hist(df["floor_"+col], label="floor_"+col, alpha=0.5, edgecolor="black", ls="dashed")
            plt.hist(df["ceiling_"+col], label="ceiling_"+col, alpha=0.3, edgecolor="black", ls="dotted")
            plt.legend()
            plt.title(col)
            plt.xlabel("degrees")
            plt.savefig(f"./plots/{col}.png")
        plt.close()

        # corr_matrix.png
        cols = df.columns[3:]
        corr_matrix = df[cols].corr()
        # matrix = np.triu(corr_matrix)
        plt.figure(figsize=(11, 6))
        sns.heatmap(
            data=corr_matrix, 
            annot=True, 
            # mask=matrix
        )
        plt.savefig('./plots/corr_matrix.png')
        plt.close()

        # scatterplot.png
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x="floor_max", y="floor_mean", hue="gt_corners", palette="Set2")
        plt.savefig('./plots/scatterplot.png')
        plt.clf()

        # bars.png
        gt_dict = df["gt_corners"].value_counts().to_dict()
        rb_dict = df["rb_corners"].value_counts().to_dict()

        gt_nums, gt_counts = gt_dict.keys(), gt_dict.values()
        rb_nums, rb_counts = rb_dict.keys(), rb_dict.values()

        _, ax = plt.subplots(1, 2, figsize=(12, 5))
        bar_container0 = ax[0].bar(gt_nums, gt_counts)
        bar_container1 = ax[1].bar(rb_nums, rb_counts)

        ax[0].set(ylabel='rooms', title='gt values count', ylim=(0, 1300))
        ax[0].bar_label(bar_container0)

        ax[1].set(ylabel='rooms', title='rb values count', ylim=(0, 1300))
        ax[1].bar_label(bar_container1)
        plt.savefig('./plots/bars.png')
        plt.close()
        plt.clf()

        # mean_dependency.png
        plt.scatter(x=df["floor_mean"], y=df["ceiling_mean"], color="red")
        plt.xlabel("floor_mean")
        plt.ylabel("ceiling_mean")
        plt.title("Dependence of ceiling_mean on floor_mean")
        plt.savefig('./plots/dependence.png')
        plt.close()
        plt.clf()

        # pairplor.png
        sns.pairplot(df)
        plt.savefig("./plots/pairplot.png")
        plt.close()
        plt.clf()
        


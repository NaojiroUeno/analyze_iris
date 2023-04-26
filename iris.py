import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, ward
from scipy.special import comb
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import NMF, PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, Normalizer, RobustScaler, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, plot_tree

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

class AnalyzeIris:
    def __init__(self) -> None:
        dataset = load_iris()  # irisデータセット
        self.__label = "iris"
        self.random_state = 0
        self.target_names = dataset.target_names
        self.target = set(dataset.target)
        self.feature_names = dataset.feature_names

        self.df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        self.df["label"] = dataset.target
        self.test_score_df = pd.DataFrame()
        self.kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        self.dtc = DecisionTreeClassifier()
        self.rfc = RandomForestClassifier()
        self.gbc = GradientBoostingClassifier()
        self.marker_list = ["o", "^", "v", "<", ">", "*", ".", ","]  # プロット時のマーカー
        # self.data = self.df.loc[:, self.df.columns != 'label'].values

    # 練習問題1

    def get(self):
        """get dataframe

        Returns:
            DataFrame: iris data
        """
        return self.df

    def get_correlation(self):
        """print correlation info

        Args:
          void
        Returns:
          print correlations
        """
        corr_df = self.df.drop(columns="label")
        return corr_df.corr()

    def pair_plot(self, **args) -> None:
        """pair plot

        Args:
          void
        Returns:
          print graphs
        """
        df = sns.load_dataset(self.__label)
        if len(args) > 0:
            sns.pairplot(df, hue="species", diag_kind=args["diag_kind"])
        else:
            sns.pairplot(df, hue="species", diag_kind="hist")
        plt.show()

    # 練習問題2
    def all_supervised(self, n_neighbors):
        """supervise all model

        Args:
            n_neighbors (int):  determine numbers of sample
        Returns:
            print all model's score
        """
        models = {
            "LogisticRegression": LogisticRegression(),
            "LinearSVC": LinearSVC(),
            "SVC": SVC(),
            "DecisionTreeClassifier": self.dtc,
            "KNeighborsClassifier": KNeighborsClassifier(n_neighbors),
            "LinearRegression": LinearRegression(),
            "RandomForestClassifier": self.rfc,
            "GradientBoostingClassifier": self.gbc,
            "MLPClassifier": MLPClassifier(),
        }

        rawData = self.df.loc[:, self.df.columns != "label"].values

        for name, model in models.items():
            print("=== {} ===".format(name))
            tmp = []
            for i_train, i_test in self.kf.split(self.df):
                X_train, X_test = rawData[i_train], rawData[i_test]
                y_train, y_test = self.df["label"][i_train], self.df["label"][i_test]
                model = model.fit(X_train, y_train)
                print("test score: {:.3f} ".format(model.score(X_test, y_test)), end="")
                print("train score: {:.3f}".format(model.score(X_train, y_train)))
                tmp.append(model.score(X_test, y_test))
            self.test_score_df[name] = tmp

    def get_supervised(self):
        """supervise all model by table

        Args:
          void
        Returns:
          DataFrame
        """
        return self.test_score_df

    def best_supervised(self) -> float:
        """select best method

        Args:
          void
        Returns:
          best_method, best_score
        """
        mean = self.test_score_df.mean(numeric_only=True)
        best_score = 0
        best_method = ""
        for model, params in mean.items():
            if params > best_score:
                best_score = params
                best_method = model

        return best_method, best_score

    def plot_feature_importances_all(self) -> None:
        """draw feature's importances"""
        trees = {
            "DecisionTreeClassifier": self.dtc,
            "RandomForestClassifier": self.rfc,
            "GradientBoostingClassifier": self.rfc,
        }
        n_features = len(self.feature_names)

        for model_name, model in trees.items():
            plt.barh(range(n_features), model.feature_importances_, align="center")
            plt.yticks(np.arange(n_features), self.feature_names)
            plt.xlabel("Feature inportance:{}".format(model_name))
            plt.show()

    def visualize_decision_tree(self) -> None:
        """draw tree graph"""
        plt.figure(figsize=(10, 8))
        plot_tree(
            self.dtc,
            feature_names=self.feature_names,
            class_names=self.target_names,
            max_depth=7,
            filled=True,
            fontsize=10,
        )
        plt.show()

    def _get_pattern(self, num) -> list:
        """return feature pattern
        Args:
           num (int): dataset pattern num
        Returns:
           list: combination of feature's number
        """
        pattern = []
        for i in range(num):
            tmp = []
            for j in range(i + 1, num):
                tmp = [i, j]
                pattern.append(tmp)
        return pattern

    # 練習問題3
    def plot_scaled_data(self) -> None:
        """plot train data and test data"""
        rawData = self.df.loc[:, self.df.columns != "label"].values
        scalers = {
            "MinMaxScaler": MinMaxScaler(),
            "StandardScaler": StandardScaler(),
            "RobustScaler": RobustScaler(),
            "Normalizer": Normalizer(),
        }

        model = LinearSVC()

        # 0:sepal length (cm) 1:sepal width (cm) 2:petal length (cm) 3:petal width (cm)
        num = comb(len(self.feature_names), 2, exact=True)
        # pattern from private method "_get_pattern"
        pattern = self._get_pattern(len(self.feature_names))

        for i_train, i_test in self.kf.split(self.df):
            fig, ax = plt.subplots(6, 5, figsize=(10, 20))
            plt.subplots_adjust(wspace=1.0, hspace=1.0)
            # Original
            X_train, X_test = rawData[i_train], rawData[i_test]
            y_train, y_test = self.df["label"][i_train], self.df["label"][i_test]
            model.fit(X_train, y_train)
            print("{0:15}:".format("Original"), end="")
            print(
                "test score: {:.3f}      ".format(model.score(X_test, y_test)), end=""
            )
            print("train score: {:.3f}".format(model.score(X_train, y_train)))

            for i in range(num):
                row = i
                col = 0
                x_col_index, y_col_index = pattern[i]
                train_x_axis, train_y_axis = (
                    X_train[:, x_col_index],
                    X_train[:, y_col_index],
                )
                test_x_axis, test_y_axis = (
                    X_test[:, x_col_index],
                    X_test[:, y_col_index],
                )

                ax[row, col].set_title("Original", size=10)
                ax[row][col].scatter(train_x_axis, train_y_axis, color="blue", s=10)
                ax[row][col].scatter(
                    test_x_axis, test_y_axis, color="red", marker="^", s=10
                )
                ax[row][col].set_xlabel(self.feature_names[x_col_index], size=8)
                ax[row][col].set_ylabel(self.feature_names[y_col_index], size=8)

            # each scaler
            count = 1
            for name, scaler in scalers.items():
                scaler.fit(X_train)
                X_train_scaled = scaler.transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                model.fit(X_train_scaled, y_train)
                print("{0:15}:".format(name), end="")
                print(
                    "test score: {:.3f}      ".format(
                        model.score(X_test_scaled, y_test)
                    ),
                    end="",
                )
                print(
                    "train score: {:.3f}".format(model.score(X_train_scaled, y_train))
                )
                for index in range(num):
                    row = index
                    col = count
                    x_col_index, y_col_index = pattern[index]
                    train_x_axis, train_y_axis = X_train_scaled[:, x_col_index], X_train_scaled[:, y_col_index]
                    test_x_axis, test_y_axis = X_test_scaled[:, x_col_index], X_test_scaled[:, y_col_index]

                    ax[row, col].set_title(name, size=10)
                    ax[row][col].scatter(train_x_axis, train_y_axis, color="blue", s=10)
                    ax[row][col].scatter(test_x_axis, test_y_axis, color="red", marker="^", s=10)
                    ax[row][col].set_xlabel(self.feature_names[x_col_index], size=8)
                    ax[row][col].set_ylabel(self.feature_names[y_col_index], size=8)
                count += 1

            plt.show()
            print("=========================================================================")
    def _plot_data(self, data, label, scaler, featureAnarysis) -> None:
        """method for plot data point

        Args:
            data (numpy.ndarray): data from dataset
            label (pandas.core.series): label from dataset
            scaler (sklearn.preprocessing): for data scaling
            featureAnarysis (sklearn.decomposision): for feature anarysis
        """
        plt.figure(figsize=(8, 6))

        for index in range(len(self.target_names)):
            classData = data[label == index]
            if scaler != None:
                classData_scaled = scaler.transform(classData)
            else:
                classData_scaled = classData
            classData_featureAnarysis = featureAnarysis.transform(classData_scaled)
            plt.scatter(
                classData_featureAnarysis[:, 0], classData_featureAnarysis[:, 1], marker=self.marker_list[index % 9]
            )

        plt.legend(self.target_names, loc=4)
        plt.xlabel("First component")
        plt.ylabel("Second component")
        plt.matshow(featureAnarysis.components_, cmap="viridis")
        plt.yticks([0, 1], ["First component", "Second component"])
        plt.colorbar()
        plt.xticks(
            range(len(self.feature_names)), self.feature_names, rotation=60, ha="left"
        )
        plt.xlabel("Feature")
        plt.ylabel("MMF components")
        plt.show()

    def plot_pca(self, n_components):
        rawData = self.df.loc[:, self.df.columns != "label"].values
        label = self.df["label"]

        scaler = StandardScaler()
        scaler.fit(rawData)
        X_scaled = scaler.transform(rawData)

        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)

        self._plot_data(rawData, label, scaler, pca)

        X_scaled = pd.DataFrame(
            X_scaled,
            columns=[
                "sepal length (cm)",
                "sepal width (cm)",
                "petal length (cm)",
                "petal width (cm)",
            ],
        )

        return X_scaled, pca

    def plot_nmf(self, n_components) -> None:
        """plot nmf graph

        Args:
           n_components (int): num of components
        """
        rawData = self.df.loc[:, self.df.columns != "label"].values
        label = self.df["label"]

        nmf = NMF(n_components=n_components, random_state=self.random_state)
        nmf.fit(rawData)

        self._plot_data(rawData, label, None, nmf)

    def plot_tsne(self) -> None:
        """plot tsne graph"""
        rawData = self.df.loc[:, self.df.columns != "label"].values
        label = self.df["label"]

        tsne = TSNE(random_state=self.random_state)
        data_tsne = tsne.fit_transform(rawData)
        plt.figure(figsize=(8, 6))
        plt.xlim(data_tsne[:, 0].min(), data_tsne[:, 0].max() + 1)
        plt.ylim(data_tsne[:, 1].min(), data_tsne[:, 1].max() + 1)

        for i in range(len(rawData)):
            plt.text(
                data_tsne[i, 0],
                data_tsne[i, 1],
                str(label[i]),
                fontdict={"weight": "bold", "size": 9},
            )

        plt.xlabel("t-SNE feature 0")
        plt.ylabel("t-SNE feature 1")

        plt.show()

    def plot_k_means(self) -> None:
        """compare kmeans and original"""
        print("test")
        rawData = self.df.loc[:, self.df.columns != "label"].values
        label = self.df["label"]
        scaler = StandardScaler()
        scaler.fit(rawData)
        X_scaled = scaler.transform(rawData)

        pca = PCA(n_components=2)
        pca.fit(X_scaled)

        kmeans = KMeans(n_clusters=3)
        kmeans.fit(X_scaled)
        label_kmeans = kmeans.predict(X_scaled)
        print("KMeans法で予測したラベル:\n{}".format(label_kmeans))
        plt.figure(figsize=(8, 6))
        color = ["green", "red", "blue"]

        for i in range(len(self.target_names)):
            classData = rawData[label_kmeans == i]
            classData_scaled = scaler.transform(classData)
            classData_pca = pca.transform(classData_scaled)
            plt.scatter(
                classData_pca[:, 0],
                classData_pca[:, 1],
                marker=self.marker_list[i % 9],
                color=color[i % 3],
            )

        center_pca = pca.transform(kmeans.cluster_centers_)
        for i in range(len(self.target_names)):
            plt.plot(
                center_pca[i, 0],
                center_pca[i, 1],
                marker="*",
                color="black",
                markersize=20,
            )

        plt.xlabel("First principal component")
        plt.ylabel("Second principal component")
        plt.show()

        # クラスタリング前
        plt.figure(figsize=(8, 6))
        print("実際のラベル:")
        print(np.array(label))
        for i in range(len(self.target_names)):
            classData = rawData[label == i]
            classData_scaled = scaler.transform(classData)
            classData_pca = pca.transform(classData_scaled)
            plt.scatter(
                classData_pca[:, 0],
                classData_pca[:, 1],
                marker=self.marker_list[i % 9],
                color=color[i % 3],
            )

        for i in range(len(self.target_names)):
            plt.plot(
                center_pca[i, 0],
                center_pca[i, 1],
                marker="*",
                color="black",
                markersize=20,
            )
        center_pca = pca.transform(kmeans.cluster_centers_)
        plt.xlabel("First principal component")
        plt.ylabel("Second principal component")

    def plot_dendrogram(self, **kword) -> None:
        """print dendrogram graph
        Args:
          0 or 1(truncate)
        """
        rawData = self.df.loc[:, self.df.columns != "label"].values
        linkage_array = ward(rawData)
        if len(kword) > 0:
            dendrogram(linkage_array, truncate_mode="lastp", p=10)
            plt.show()
        else:
            dendrogram(linkage_array)
            plt.show()

    def plot_dbscan(self, **kword) -> None:
        """plot dbscan data"""
        rawData = self.df.loc[:, self.df.columns != "label"].values
        label = self.df["label"]
        plt.figure(figsize=(8, 6))
        if len(kword) > 0:
            dbscan = DBSCAN(eps=kword["eps"], min_samples=kword["min_samples"])
            if kword["scaling"]:
                scaler = StandardScaler()
                scaler.fit(rawData)
                X_scaled = scaler.transform(rawData)
                clusters = dbscan.fit_predict(X_scaled)
                clusters_kind = list(set(clusters))
                print(clusters)
                for i in clusters_kind:
                    classData = rawData[clusters == i]
                    classData_scaled = scaler.transform(classData)
                    plt.scatter(classData_scaled[:, 2], classData_scaled[:, 3])
        else:
            dbscan = DBSCAN()
            clusters = dbscan.fit_predict(rawData)
            clusters_kind = list(set(clusters))
            print(clusters)
            for i in clusters_kind:
                classData = rawData[clusters == i]
                plt.scatter(classData[:, 2], classData[:, 3])
        plt.xlabel("Feature2")
        plt.ylabel("Feature3")
        plt.show()

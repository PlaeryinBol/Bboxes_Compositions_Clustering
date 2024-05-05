import os
import shutil
import sys

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import scipy.cluster.hierarchy as sch
import timm
from PIL import Image
from sklearn import metrics
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import config
import utils


class Clusterizator():
    def __init__(self):
        self.data_df = pd.read_table(config.DATA_DF)
        self.unique_files = self.data_df['file'].unique().tolist()
        self.unique_classes = self.data_df['cls'].unique().tolist()
        self.image_sectors = utils.create_sectors()

    def create_custom_features_df(self, drop_duplicates: bool = True) -> pd.DataFrame:
        """Create df with custom features for every unique image file."""
        columns = ['file', 'image_occupancy'] + \
                  [f'{cls}_cls_count' for cls in self.unique_classes] + \
                  [f'mean_{cls}_cls_width' for cls in self.unique_classes] + \
                  [f'mean_{cls}_cls_height' for cls in self.unique_classes] + \
                  [f'{cls}_cls_depth_count' for cls in self.unique_classes] + \
                  [f'identical_{cls}_cls' for cls in self.unique_classes] + \
                  [f'mean_{cls}_cls_centers_dist' for cls in self.unique_classes] + \
                  [f'sector_{s}_{cls}_cls_occupancy' for s, _ in enumerate(self.image_sectors)
                   for cls in self.unique_classes]

        df = pd.DataFrame(columns=columns)
        df['file'] = self.unique_files
        rows_list = []

        for _, file in enumerate(tqdm(self.unique_files, desc='Create custom features')):
            row_data = {'file': file}
            file_row = self.data_df.loc[self.data_df['file'] == file]

            image_occupancy = utils.calculate_image_occupancy(file)
            row_data['image_occupancy'] = image_occupancy

            cls_data = {}
            # get features separately for every class
            for cls in self.unique_classes:
                cls_data[cls] = file_row.loc[file_row['cls'] == cls].to_numpy()[:, 1:]
                row_data[f'{cls}_cls_count'] = len(cls_data[cls])
                row_data[f'{cls}_cls_depth_count'] = len(np.unique(cls_data[cls][:, 5]))

                # check that the picture contains a bbox of the current class
                if cls_data[cls].any():
                    row_data[f'mean_{cls}_cls_width'] = np.mean(cls_data[cls][:, 2] - cls_data[cls][:, 0])
                    row_data[f'mean_{cls}_cls_height'] = np.mean(cls_data[cls][:, 3] - cls_data[cls][:, 1])
                    row_data[f'identical_{cls}_cls'] = len(np.unique(utils.box_area(cls_data[cls][:, :4]))) / len(cls_data[cls])
                    row_data[f'mean_{cls}_cls_centers_dist'] = utils.get_mean_center_distance(cls_data[cls])
                else:
                    row_data[f'mean_{cls}_cls_width'] = 0
                    row_data[f'mean_{cls}_cls_height'] = 0
                    row_data[f'identical_{cls}_cls'] = 0
                    row_data[f'mean_{cls}_cls_centers_dist'] = 0

                # calculate the coverage of each sector by bboxes of the current class
                for s, sector in enumerate(self.image_sectors):
                    row_data[f'sector_{s}_{cls}_cls_occupancy'] = np.sum(utils.box_intersetion(cls_data[cls][:, :4], sector)) / config.SECTOR_AREA
            rows_list.append(row_data)

        df = pd.DataFrame(rows_list)
        # drob dublicates (files with identical content) from df
        if drop_duplicates:
            df = df.drop_duplicates(subset=df.columns[1:]).reset_index(drop=True)
        return df

    def create_image_features_df(self, df: pd.DataFrame, embedding_dim: int = 768) -> pd.DataFrame:
        """Create df with image features for every unique image file."""
        model = timm.create_model(config.IMG_EMBEDDER_MODEL, pretrained=True, num_classes=0).eval().to('cuda')
        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)

        rows_list = []
        for i, file in enumerate(tqdm(df['file'], desc='Create image features')):
            image = Image.open(os.path.join(config.DATASET_PATH, file)).convert("RGB")
            image_tensor = transforms(image).unsqueeze(0).to('cuda')
            output = model(image_tensor).tolist()[0]
            rows_list.append(output)

        embedding_df = pd.DataFrame(rows_list, columns=[str(i) for i in range(embedding_dim)])
        return embedding_df

    def get_features_df(self) -> pd.DataFrame:
        """Create/read from disk features df."""
        if os.path.exists(config.FEATURES_DF_PATH):
            features_df = pd.read_table(config.FEATURES_DF_PATH)
        else:
            features_df = self.create_custom_features_df()
            # add image features to df if necesary
            if config.USE_IMAGE_FEATURES:
                image_features = self.create_image_features_df(features_df)
                features_df = pd.concat([features_df, image_features.reset_index(drop=True)], axis=1)

            features_df.to_csv(config.FEATURES_DF_PATH, sep='\t', index=False)
        return features_df

    def apply_pca(self, df: pd.DataFrame) -> None | pd.DataFrame:
        """Apply PCA for features df."""
        X = df.loc[:, df.columns[1:]]
        pipe = Pipeline([('scaler', StandardScaler())])
        X_std = pd.DataFrame(pipe.fit_transform(X))

        # if the config.PCA_COMPONENTS value is not specified, draw a graph to determine it
        if not config.PCA_COMPONENTS:
            pca = PCA().fit(X_std)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(1, len(X_std.columns) + 1)), y=pca.explained_variance_ratio_.cumsum(),
                                     mode='lines+markers', name='Explained Variance'))
            fig.update_layout(title='Explained Variance by Components', xaxis_title='Number of components',
                              yaxis_title='Cumulative Explained variance')
            fig.write_image("./plots/explained_variance.png")
            print("""Find from the graph how many components need to be taken in order to maintain the variance
                  at a sufficient level (0.8+), then set it to config.PCA_COMPONENTS""")
            sys.exit()
        else:
            pca = PCA(config.PCA_COMPONENTS)
            X = pca.fit_transform(X_std)
            important_features_per_component = [np.argsort(np.abs(pca.components_[i]))[::-1][:5]
                                                for i in range(len(pca.components_))]
            feature_names = [df.columns[1:][indices].tolist() for indices in important_features_per_component]
            print(f"Most important features: {feature_names[0]}")
            pca_df = pd.DataFrame(X, columns=[f'component_{c}' for c in range(1, config.PCA_COMPONENTS + 1)])
            return pca_df

    def find_optimal_clusters_count(self, df: pd.DataFrame, max_clusters: int = 20) -> None:
        """Draw a wcss plot and hierarchical dendrogram for determining optimal clusters count."""
        # if config.PREDEFINED_CLUSTERS_COUNT already specified, nothing is drawn
        if config.PREDEFINED_CLUSTERS_COUNT:
            return

        wcss = []
        for i in tqdm(range(1, max_clusters + 1)):
            kmeans_pca = KMeans(n_clusters=i)
            kmeans_pca.fit(df)
            wcss.append(kmeans_pca.inertia_)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(1, max_clusters + 1)), y=wcss, mode='lines+markers'))
        fig.update_layout(title='KMeans with WCSS', xaxis_title='Number of clusters', yaxis_title='WCSS')
        fig.write_image("./plots/wcss.png")
        print('Find the elbow on a graph, then set it to config.PREDEFINED_CLUSTERS_COUNT')

        fig_1 = ff.create_dendrogram(sch.linkage(df, method='ward'), orientation='bottom')
        fig_1.update_layout(title='Hierarchical dendrogram', xaxis_title='Samples', yaxis_title='Euclidean distances')
        fig_1.update_xaxes(showticklabels=False)
        fig_1.write_image("./plots/hierarchical_dendrogram.png")
        print('Find the optimal number of clusters using dendrogram, then set it to config.PREDEFINED_CLUSTERS_COUNT')
        sys.exit()

    def find_optimal_eps(self, df: pd.DataFrame, eps_interval: tuple = (0.5, 10, 0.5)) -> None:
        """Draw plot for silhouette scores and clusters count for determining optimal eps/max eps for DBSCAN/OPTICS."""
        # if config.EPS already specified, nothing is drawn
        if config.EPS:
            return

        parameters = np.arange(*eps_interval)
        # specified parameters grid for eps
        if config.CLUSTERING_TYPE == 'dbscan':
            parameter_grid = ParameterGrid({'eps': parameters})
            model = DBSCAN(min_samples=config.MIN_SAMPLES, n_jobs=-1)
        elif config.CLUSTERING_TYPE == 'optics':
            parameter_grid = ParameterGrid({'max_eps': parameters})
            model = OPTICS(min_samples=config.MIN_SAMPLES, n_jobs=-1)

        silhouette_scores, clusters_count = [], []
        # evaluation based on silhouette_score
        for p in tqdm(parameter_grid, desc='Search best eps/max_eps'):
            model.set_params(**p)
            model.fit(df)
            cl_count = len(np.unique(model.labels_))

            # stop if we have to few clusters
            if cl_count < 2:
                continue

            ss = metrics.silhouette_score(df, model.labels_)
            silhouette_scores += [ss]
            clusters_count += [cl_count]

        # bring both values to the same scale
        silhouette_scores = StandardScaler().fit_transform(np.array(silhouette_scores).reshape(-1, 1)).flatten()
        clusters_count = StandardScaler().fit_transform(np.array(clusters_count).reshape(-1, 1)).flatten()

        trace1 = go.Scatter(
            x=parameters,
            y=silhouette_scores,
            mode='lines',
            name='std silhouette scores',
            line=dict(color='red')
        )

        trace2 = go.Scatter(
            x=parameters,
            y=clusters_count,
            mode='lines',
            name='std clusters count',
            line=dict(color='blue')
        )

        layout = go.Layout(
            title=f'silhouette scores and clusters count for different {config.CLUSTERING_TYPE} eps',
            xaxis=dict(title='eps', tickvals=parameters),
            yaxis=dict(title='std score'),
            showlegend=True
        )

        fig = go.Figure(data=[trace1, trace2], layout=layout)
        fig.write_image(f"./plots/optimal_eps_search_for_{config.CLUSTERING_TYPE}.png")
        print('Find the optimal eps by plot, then set it to config.EPS')
        sys.exit()

    def apply_clustering(self, df: pd.DataFrame, type: str) -> np.array:
        """Apply clustering, counting the silhouette score."""
        match type:
            case "kmeans":
                self.find_optimal_clusters_count(df)
                model = KMeans(n_clusters=config.PREDEFINED_CLUSTERS_COUNT, random_state=42).fit(df)
            case "dbscan":
                self.find_optimal_eps(df)
                model = DBSCAN(eps=config.EPS, min_samples=config.MIN_SAMPLES, n_jobs=-1).fit(df)
            case "optics":
                self.find_optimal_eps(df)
                model = OPTICS(max_eps=config.EPS, min_samples=config.MIN_SAMPLES, n_jobs=-1).fit(df)

        clusters = model.labels_
        silhouette_score = metrics.silhouette_score(df, clusters)
        print(f'silhouette_score: {silhouette_score:.2f}')
        return clusters

    def plot_and_save_clusters(self, df: pd.DataFrame) -> None:
        """Draw clusters on plot, saving images to separate folders if necessary."""
        counts = df['cluster'].value_counts().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=counts['cluster'], y=counts['count']))
        fig.update_layout(title=f'Clusters distribution for {config.CLUSTERING_TYPE}', yaxis_title='Elements',
                          xaxis={'title': 'Cluster idx', 'tickvals': counts['cluster']})
        fig.write_image(f"./plots/clusters_distribution_for_{config.CLUSTERING_TYPE}.png")

        if not config.SAVE_CLUSTERS:
            return

        if not os.path.exists(config.CLUSTERS_DIR):
            os.makedirs(config.CLUSTERS_DIR)
        else:
            shutil.rmtree(config.CLUSTERS_DIR)

        # saving clustered images in separate folders
        for i in df['cluster'].unique():
            subfolder_path = os.path.join(config.CLUSTERS_DIR, str(i))
            os.makedirs(subfolder_path)
            target_files = df.loc[df['cluster'] == i, 'file']
            for f in target_files:
                shutil.copy(os.path.join(config.DATASET_PATH, f), os.path.join(subfolder_path, f))


if __name__ == "__main__":
    clusterizator = Clusterizator()
    features_df = clusterizator.get_features_df()
    X = clusterizator.apply_pca(features_df)
    clusters = clusterizator.apply_clustering(X, config.CLUSTERING_TYPE)
    cluster_df = pd.DataFrame({'file': features_df['file'], 'cluster': clusters})
    clusterizator.plot_and_save_clusters(cluster_df)

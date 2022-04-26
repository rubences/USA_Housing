def detect_outliners(df,features):
 outliner_indices = []
 for c in features:
 # 1st quartile
 Q1 = np.percentile(df[c],25)
 # 3rd quartile
 Q3 = np.percentile(df[c],75)
 # IQR
 IQR = Q3 - Q1
 # Outlier step
 outlier_step = IQR * 1.5
 # detect outlier and their indeces
 outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
 # store indeces
 outlier_indices.extend(outlier_list_col)
 outliner_indices = Counter(outlier_indices)
 multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
 return multiple_outliers
USA_Housing.loc[detect_outliners(USA_Housing,["Avg. Area House Age","Sibsip","Address","Avg. Area Income"])]
USA_Housing_len = len(USA_Housing)
USA_Housing.head()
sns.heatmap(USA_Housing.isnull(),
 yticklabels=False,
 cbar=False,
 cmap='magma')
plt.title('Valores perdidos en conjunto de train')
plt.xticks(rotation=90)
plt.show()
USA_Housing.columns[USA_Housing.isnull().any()]
Index(['Avg. Area House Age', 'Address', 'Address'], dtype='object')
USA_Housing.isnull().sum()
USA_Housing[USA_Housing["Address"].isnull()]
USA_Housing.boxplot(column="Avg. Area Income",by = "Address")
plt.show()
USA_Housing["Address"] = USA_Housing["Address"].fillna("c")
USA_Housing[USA_Housing["Address"].isnull()]
USA_Housing[USA_Housing["Avg. Area Income"].isnull()]
USA_Housing["Avg. Area Income"] = USA_Housing["Avg. Area Income"].fillna(np.meanUSA_Housing[USA_Housing["Address"] == 3]["Avg. Area Income"]))
USA_Housing[USA_Housing["Avg. Area Income"].isnull()]
corr = USA_Housing.corr()
f,ax = plt.subplots(figsize=(9,6))
sns.heatmap(corr, annot = True, linewidths=1.5 , fmt = '.2f',ax=ax)
plt.show()
g = sns.factorplot(x = "Address", y = "Address", data = USA_Housing, kind = "bar", size = 6)
g.set_ylabels("Probabilidad de supervivencia")
plt.show()
g = sns.factorplot(x = "Address", y = "Address",kind = "bar", data = USA_Housing, size = 6)
g.set_ylabels("Probabilidad de supervivencia")
plt.show()
g = sns.FacetGrid(USA_Housing, col = "Address")
g.map(sns.displot, "Avg. Area House Age", bins = 25)
plt.show()
g = sns.FacetGrid(USA_Housing, row = "Address", col = "Address", size = 2.3)
g.map(sns.barplot, "Address", "Avg. Area Income")
g.add_legend()
plt.show()
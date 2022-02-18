#!/usr/bin/env python
# coding: utf-8

# # Rendu du TP2 M2 ENSP

# ## Auteur : Soumaïla DA    | Professeur : Vasseur Corentin

# # Partie 1

# # Exercice 1

# ### SciPy
#     Librairie python permettant d'utiliser un ensemble d'algorithmes codés en c

# ### Numpy
#     Librairie python pour manipuler les listes de manière optimisé (codé c)

# ### Pandas
#     Librairie python pour manipuler des données sous forme de dataFrame

# ### Matplotlib
#     Librairie python pour les graphiques

# ### Seaborn
#     Librairie python pour les graphique, un complément à matplotlib

# ### scikit-learn
#     Librairie python de Machine learning

# ### Tensorflow
#     Librairie python de Deep Learning

# # Exercice 2

# In[1]:


import numpy as np


# ### Créer les vecteurs a et b

# In[2]:


a = np.array([1,2,3])
b = np.array([4,5,6])


# ### Sommer et soustraire a et b.

# In[4]:


np.add(a,b) # Somme avec add
np.subtract(a,b) # Soustraction avec subtract


# ### Ajouter 10 aux deux vecteurs.

# In[5]:


# La fonction insert du module numpy
a = np.insert(a, 3, 10)
b = np.insert(b, 3, 10)


# ### Comparer les 2 vecteurs

# In[6]:


# La fonction array_equal
np.array_equal(a, b)


# ### Concatener les deux vecteurs

# In[7]:


# La fonction concactenate
np.concatenate((a,b),axis=0)


# ### Tracer la fonction x² dans l’intervalle [-10;10]

# In[8]:


import matplotlib.pyplot as plt


# In[10]:


# Géner un échantillon de 100 nombre compris entre -10 et 10
x = np.linspace(-10, 10, num = 100)
x.shape


# In[13]:


# Représentation de la fonction x²
plt.figure(figsize = (8, 4))
plt.plot(x,(x**2), label = 'y = -2X') # Tracer de la courbe
plt.xlabel('x')
plt.ylabel('y')
plt.title("Fonction x² sur [-10, 10]")
plt.grid(True)
plt.show() # Afficher le graphe


# ### Tracer les fonctions 2x² + 3 et -x² sur le même graphe

# In[15]:


# Tracer de 2x² + 3 et -x²
plt.figure(figsize=(8, 5))
plt.plot(x,(2 * x**2 + 3), label = 'y1 = 2X² + 3',ls = '--') # Tracer de la courbe 1
plt.plot(x,(-x**2), 'o-', label = 'y1 = -2X') # Tracer de la courbe 2
plt.xlabel('x')
plt.ylabel('y')
plt.title("Tracer plusieurs fonctions sur le même graphe")
plt.legend() # Afficher la légende
plt.grid(True)
plt.show() # Afficher le graphe


# ### Tracer sur deux axes différents

# In[20]:


fig, ax = plt.subplots(1,2, figsize=(10, 5))
ax[0].plot(x,(2 * x**2 + 3), label = 'y1 = 2X² + 3',ls = '--') # Tracer de la courbe 1
ax[1].plot(x,(-x**2), '--', label = 'y1 = -2X', c = 'red') # Tracer de la courbe 2
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[0].set_title("2x² + 3")

ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
ax[1].set_title("-x²")


# # Partie 2 : Manipulation des données

# # Exercice 3 : Manipulation de fichier avec 'open'

# ### Lecture du fichier Exercice 3.csv

# In[21]:


# Ouverture du fichier Exercice 3 en mode lecture ('r' : read)
with open('C:/Users/SOUMAILA DA/Desktop/Python/Exercice 3.csv','r') as Pays :
    for line in Pays :
        print(line)


# ### Ecrire dans le fichier Exercice 3.csv

# In[25]:


# Création d'une nouvelle liste
Bresil = ["4", "Brésil", "Brasilia", "207847528"]

# Ouverture du fichier Exercice 3 en mode écriture ('w' : write)
with open('C:/Users/SOUMAILA DA/Desktop/Python/Exercice 3.csv','w') as Pays :
     # Parcourir la liste Bresil avec for
    for line in Bresil :
        # écrire chaque élément de la liste Bresil dans le fichier csv
        Pays.write(f"{line}\n")

# Rélire le fichier
with open('C:/Users/SOUMAILA DA/Desktop/Python/Exercice 3.csv','r') as Pays :
    for line in Pays :
        print(line)


# # Manipulation de données avec Pandas

# # Exercice 4 : Series et DataFrame

# In[26]:


import pandas as pd


# ### Créer une série pandas à l’aide de : pd.Series

# In[27]:


# Créer la série Pays
Pays = pd.Series(['France','Belgique','Inde','Brésil'], index = [1,2,3,4])

# Afficher la série
Pays


# In[28]:


# Afficher les valeurs de la série
Pays.values


# In[29]:


# Afficher les index de la série
Pays.index


# In[30]:


# Afficher les éléments 1 et 3 de la série
Pays[[1,3]]


# ### Définir la série à partir d'un dictionnaire

# In[31]:


# Définir le dictionnaire Population
Populations = { 'Paris': 2187526, 'Marseille': 863310, 'Lyon': 516092,'Toulouse': 479553, 'Nice': 340017 }
Populations


# In[32]:


# Définir le dictionnaire Superficies
Superficies = { 'Paris': 105.4, 'Marseille': 240.6, 'Lyon': 47.87,'Toulouse': 118.3, 'Nice': 71.92 }
Superficies


# In[33]:


# Transformation du dictionnaire en Série
Populations = pd.Series(Populations)
Populations


# In[34]:


# Transformation du dictionnaire en Série
Superficies = pd.Series(Superficies)
Superficies


# ### Créer un Dataframe à partir des deux séries populations et superficie

# In[35]:


Demographie = pd.DataFrame({'Populations' : Populations, 'Superficies' : Superficies})
Demographie


# # Exercice 5 : Chargement d’un jeu de données

# ### Charger les données avec Pandas.

# In[36]:


# Données d'hopitalisation COVID du 04-02-2022 en France
Hospi = pd.read_csv('C:/Users/SOUMAILA DA/Desktop/Python/donnees-hospitalieres-covid19-2022-02-04-19h10.csv', sep = ';')
Hospi


# In[40]:


# Selectionner les colonnes dep, sexe, jour, hosp, rea, rad, dc
df_hospi = Hospi[['dep', 'sexe', 'jour', 'hosp', 'rea', 'rad', 'dc']]
df_hospi.head()


# ### Calculer le nombre de personnes actuellement hospitalisées par département (04-02-2022).

# In[42]:


Hospi_4fev22 = df_hospi[(df_hospi.sexe==0) & (df_hospi.jour=='2022-02-04')][['dep','hosp']]
Hospi_4fev22.head()


# ### Calculer le nombre de personnes actuellement hospitalisées par département pour les 3 derniers jours

# In[45]:


jj3 = (df_hospi.sexe==0) & (df_hospi.jour.isin(['2022-02-04','2022-02-03','2022-02-02'])) # Filtre sur les 3 derniers jours
Hospi_3dj = df_hospi[jj3][['dep','jour','hosp']]
Hospi_3dj.head()


# ### Tracer la courbe d'évolution des départements 75, 59, 13 à l’aide de la librairie Matplotlib.

# In[49]:


# Liste de departements
depart = ['75', '59', '13']

# Selection des données sur les 3 departements pour tous les sexes
Paris_Lille_Marseil = df_hospi[df_hospi.dep.isin(depart)]
Paris_Lille_Marseil = Paris_Lille_Marseil[Paris_Lille_Marseil.sexe == 0]
Paris_Lille_Marseil.head()


# In[50]:


# Selectionner les données des 3 départements juste pour l'année 2021
An_2021 = Paris_Lille_Marseil.jour.str.contains('2021-')
Paris_Lille_Marseil_2021 = Paris_Lille_Marseil[An_2021]
Paris_Lille_Marseil_2021.head()


# In[51]:


# Tracer du Graphique
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
for dep in depart:
    Paris_Lille_Marseil_2021[Paris_Lille_Marseil_2021.dep == dep]         .plot(
            x = 'jour',
            y = 'hosp',
            ax = ax,
            label = f"dep {dep}",
            xlabel = 'jour',
            ylabel = 'nb hosp',
    )
ax.set_title("Evolution du nombre d'hosp pour les départements de Paris-Lille-Marseille en 2021")
plt.xticks(rotation = 45)
ax.grid()


# ### BONUS : jouer avec les données

# ### Nombre total des victimes du COVID-19 par département à la date du 04-02-2022

# In[55]:


Dec_4fev22 = df_hospi[(df_hospi.sexe==0) & (df_hospi.jour=='2022-02-04')][['dep','dc']]
Dec_4fev22.head()


# In[60]:


Dec_4fev22.plot(x = 'dep', y = 'dc')


# # Partie 3 : Analyse descriptive d’un jeu de données

# # Exercice 6 (BONUS)

# In[70]:


import pylab as P
import seaborn as sns


# ### Importation de la base et identification des variables

# In[61]:


df_Titanic = pd.read_csv('C:/Users/SOUMAILA DA/Desktop/Python/titanic-passengers.csv', sep = ';')
df_Titanic.head()


# ### Identifier le nombre de ligne et de colonnes

# In[64]:


df_Titanic.describe


# La base comprend 891 lignes (Passagers) et 12 colonnes (Variables)

# ### Identifier les noms des colonnes

# In[65]:


df_Titanic.columns


# ### Identifiez les variables qualitatives et les variables quantitatives.

# In[66]:


df_Titanic.dtypes


# #### Variables quantitatives:
#     • Discrètes :
#         ♣ PassengerId,
#         ♣ Survived,
#         ♣ SibSp,
#         ♣ Parch.      
#     • Continues :
#         ♣ Age
#         ♣ Fare        
#     • Textuelles :
#         ♣ Ticket
#         ♣ Name
# 
# #### Variables qualitatives :
#     • Nominales :
#         ♣ Cabin,
#         ♣ Embarked.
#     • Dichotomiques :
#         ♣ Sex.
#     • Ordinales :
#         ♣ Pclass.
# 
# La variable cible est : Survived (0 = mort, 1 = survie)

# ### Réalisez une analyse unitaire des variables

# #### ►PassengerId

# In[68]:


df_Titanic.PassengerId.describe()


# In[69]:


# Exist-il des doublons dans les identifiants ?
df_Titanic.PassengerId.duplicated().sum()


# Il n'y a pas de doublons, les identifiants vont de 1 à 891

# #### ►Survived

# In[67]:


df_Titanic["Survived"].value_counts()


# In[77]:


# Visualisation de la répartition des passagers selon la survie
sns.set(rc={'figure.figsize':(8,5)})
sns.countplot(x="Survived", data=df_Titanic)


# #### ►Pclass

# In[78]:


df_Titanic.Pclass.value_counts()


# In[81]:


# Renommer la variable
Class = {1 : 'First', 2 : 'Second', 3 : 'Third'}
df_Titanic.Pclass = [Class[item] for item in df_Titanic.Pclass]

# Visualisation de la répartition des passagers selon la classe des chambres du navire
sns.set(rc={'figure.figsize':(8,5)})
sns.countplot(x="Pclass", data=df_Titanic)


# #### ►SibSp

# In[82]:


df_Titanic.SibSp.value_counts()


# In[83]:


# Visualisation de la répartition des passagers selon le nombre de membre de la famille du passager de type frère, soeur
sns.set(rc={'figure.figsize':(8,5)})
sns.countplot(x="SibSp", data=df_Titanic)


# #### ►Name

# In[86]:


# Créer une nouvelle colonne et enregistrer les noms des famille des passagers
df_Titanic['Last Name'] = df_Titanic['Name'].str.split(", ", expand=True)[0]
df_Titanic['Last Name'].head()


# In[88]:


# Statistiques sur les noms de famille
df_Titanic['Last Name'].describe()


# Au total il y'avait 667 noms de famille différent sur le navire et le nom Andersson était le plus fréquent soit 9 passagers

# #### ►Sex

# In[89]:


df_Titanic.Sex.value_counts()


# In[90]:


# Visualisation de la répartition des passagers selon le sexe
sns.set(rc={'figure.figsize':(8,5)})
sns.countplot(x="Sex", data=df_Titanic)


# #### ►Age

# In[92]:


df_Titanic.Age.describe()


# Les passagers avaient une moyenne d'âge compris entre 29 et 30 ans, le plus âgé avait 80 ans et le plus jeune moins d'1 an (quelques mois).
# Il faut noter cependant qu'il y'a beaucoup de valeurs manquantes (177)

# In[100]:


# Visualisation de la répartition des passagers selon l'âge
sns.distplot(df_Titanic.Age.dropna()) # La fonction "dropna" permet d'ommettre les valeurs manquantes dans la répresentation 


# #### ►Parch

# In[94]:


df_Titanic.Parch.value_counts()


# In[95]:


# Visualisation de la répartition des passagers selon le nombre de membre de la famille du passager de type père, mère, fille
sns.set(rc={'figure.figsize':(8,5)})
sns.countplot(x="Parch", data=df_Titanic)


# #### ►Fare

# In[96]:


df_Titanic.Fare.describe()


# In[99]:


sns.distplot(df_Titanic.Fare.dropna())


# ### Réalisez une analyse croisée des variables quantitatives

# #### ►Survived & Age

# In[105]:


# parallèle boxplots
df_Titanic.boxplot(column="Age",by="Survived")


# On remarque que les survivants ont un âge median moins important que les non survivants. Cependant une valeur abbérante au niveau des survivants a entrainé la hausse de l'âge médian.

# #### ►Survived & Fare

# In[108]:


# Diagramme en bar de la répartition des survivants selon le prix du billet
sns.barplot(x = 'Survived', y = 'Fare', data = df_Titanic,
            order = ["Yes", "No"])
 
plt.show()


# On remarque que les passager qui ont un prix de billet plus élévé ont un taux de survie plus important que les autres

# ### Réalisez une analyse croisée des variables qualitatives.

# #### ►Survived & Sex

# In[116]:


# table de contingence
table=pd.crosstab(df_Titanic["Sex"],df_Titanic["Survived"])
print(table)


# Les femmes ont eu un taux de survie plus élévé que les hommes. En effet, 74% d'entre elles ont survécu contre seulement 18% chez les hommes.

# #### ►Survived & Pclass

# In[117]:


# table de contingence
table=pd.crosstab(df_Titanic["Pclass"],df_Titanic["Survived"])
print(table)


# Les passagers de première classe avait un taux de survie plus élévé que les autres. En effet plus de 60% des passagers de la première classe ont survécu alors que plus de 75% des passagers de la troisième classe sont décédés.

# # Exercice 7 : Nettoyer les données

# ## Pour détection des valeurs manquantes on utilise :
#     pd.isnull(), pd.notnull()

# In[128]:


df_Titanic.isnull()


# In[132]:


df_Titanic.Cabin.notnull()


# ## Pour détecter les doublons
#     df.duplicated().sum()

# In[130]:


df_Titanic.PassengerId.duplicated().sum()


# ## Comment filtrer les données manquantes ?
#     df1.dropna() pour supprimer toute ligne contenant une valeur manquante
#     df1.dropna(axis = 1) pour supprimer toute colonne contenant une valeur manquante
#     df1.dropna(how = 'all') pour supprimez les lignes manquantes
#     df1.dropna(thresh = 3) pour supprimer n'importe quelle ligne contenant 3 valeurs manquantes

# In[141]:


df_Titanic.dropna()


# In[142]:


df_Titanic.dropna(axis = 1)


# In[143]:


df_Titanic.dropna(how = 'all')


# In[144]:


df_Titanic.dropna(thresh = 3)


# ## Remplacer les données manquantes
#     df1 = df1.fillna(0) pour remplir toutes les données manquantes avec 0
#     df1.fillna('inplace = True') pour modifier sur place
#     df1.fillna({'col1' : 0, 'col2' : -1}) pour utiliser une valeur de remplissage différente pour chaque colonne
#     df1.fillna(method = 'ffill', limit = 2) pour remplir uniquement les 2 valeurs manquantes devant

# In[146]:


df1 = pd.read_csv('C:/Users/SOUMAILA DA/Desktop/Python/titanic-passengers.csv', sep = ';')


# In[147]:


# Détecter et remplacer toutes les valeurs manquantes par 'inplace = True'
df1.fillna('inplace = True')


# In[149]:


# Remplacer toutes les valeurs manquante par 0
df1.fillna(0)


# In[150]:


df1 = pd.read_csv('C:/Users/SOUMAILA DA/Desktop/Python/titanic-passengers.csv', sep = ';')


# In[152]:


# Utiliser une valeur de remplissage différente pour chaque colonne où il y'a une valeur manquante
df1.fillna({'Age' : 0, 'Cabin' : 'C52'})


# In[153]:


df1 = pd.read_csv('C:/Users/SOUMAILA DA/Desktop/Python/titanic-passengers.csv', sep = ';')


# In[154]:


#  Remplir uniquement les 2 prémières valeurs manquantes
df1.fillna(method = 'ffill', limit = 2)


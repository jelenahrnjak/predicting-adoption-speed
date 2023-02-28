import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_map_data():
    train = pd.read_csv('train.csv')
    states = pd.read_csv('data/state_labels.csv')
    colors = pd.read_csv('data/color_labels.csv')
    breeds = pd.read_csv('data/breed_labels.csv')

    train['Type'] = train["Type"].map({1: "Dog", 2: "Cat"})
    breeds['Type'] = breeds['Type'].map({1: "Dog", 2: "Cat"})

    train['AdoptionSpeed'] = train["AdoptionSpeed"].map({0: 'Same day', 1: '1-7 days', 2: '8-30 days', 3: '31-90 days',
                                                         4: 'No adoption'})
    train['Mixed'] = train.apply(lambda x: get_cross_breed(x['Breed1'], x['Breed2']), axis=1)
    train['Health'] = train["Health"].map({0: 'Not specified', 1: 'Healthy', 2: 'Minor injury', 3: 'Serious injury'})
    medical_mapping = {1: 'Yes', 2: 'No', 3: 'Not sure'}
    train['Vaccinated'] = train["Vaccinated"].map(medical_mapping)
    train['Dewormed'] = train["Dewormed"].map(medical_mapping)
    train['Sterilized'] = train["Sterilized"].map(medical_mapping)

    breed_mapping = dict(breeds[['BreedID', "BreedName"]].values)
    train["Breed1"] = train.Breed1.map(breed_mapping)
    train['Breed2'] = train.Breed2.map(breed_mapping)

    color_mapping = dict(colors[['ColorID', "ColorName"]].values)
    train["Color1"] = train.Color1.map(color_mapping)
    train['Color2'] = train.Color2.map(color_mapping)
    train['Color3'] = train.Color3.map(color_mapping)

    state_mapping = dict(states[['StateID', "StateName"]].values)
    train["State"] = train.State.map(state_mapping)
    return train


def null_values(train):
    text = "Null data:\n"
    underline = "-" * len(text)
    print(text + underline)
    print(train.isnull().sum())


def animal_types(train, palette):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    type_count = train['Type'].value_counts()

    axes[0].pie(type_count.values, colors=['#628AD0', '#EE7DC2'], labels=type_count.keys(), autopct='%1.1f%%',
                startangle=90, counterclock=False, wedgeprops={"linewidth": 2, "edgecolor": "white"})
    axes[0].set_title('Types')

    order = ['Same day', '1-7 days', '8-30 days', '31-90 days', 'No adoption']

    g = sns.countplot(x='AdoptionSpeed', hue="Type", data=train, palette=palette, order=order, ax=axes[1])
    g.set_title("Mixed animals")

    plt.subplots_adjust(right=0.75)
    text_str1 = '\n'.join([f"{k}: {v}" for k, v in type_count.items()]) + '\n\n'
    text_str = '\n'.join([f"{k}: {v}" for k, v in train.AdoptionSpeed.value_counts().items()])
    plt.text(1.05, 0.5, text_str1 + text_str,
             transform=g.transAxes, fontsize=12, ha='left', va='center')

    for p in g.patches:
        x = p.get_bbox().get_points()[:, 0]
        y = p.get_bbox().get_points()[1, 1]
        g.annotate('{:.2g}%'.format(100. * y / len(train)), (x.mean(), y), ha='center', va='bottom');

    plt.show()


def breeds_stats(train, palette):
    order = ['Same day', '1-7 days', '8-30 days', '31-90 days', 'No adoption']

    fig, axes = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [1, 2]})

    g1 = sns.countplot(x='Mixed', hue="Type", data=train, palette=palette, ax=axes[0])
    g1.set_title("Is animal mixed")
    for p in g1.patches:
        x = p.get_bbox().get_points()[:, 0]
        y = p.get_bbox().get_points()[1, 1]
        g1.annotate('{:.2g}%'.format(100. * y / len(train)), (x.mean(), y), ha='center', va='bottom')

    palette = {"Pure breed Dog": "#628AD0", "Pure breed Cat": "#EE7DC2", "Mixed Dog": "#E8C547", "Mixed Cat": "#48E865"}

    g2 = sns.countplot(x='AdoptionSpeed', hue=train["Mixed"] + ' ' + train["Type"], data=train, palette=palette,
                       ax=axes[1], order=order)
    g2.set_title("Adoption speed for mixed and pure breed animals")

    for p in g2.patches:
        x = p.get_bbox().get_points()[:, 0]
        y = p.get_bbox().get_points()[1, 1]
        g2.annotate('{:.2g}%'.format(100. * y / len(train)), (x.mean(), y), ha='center', va='bottom')

    plt.show()


def get_cross_breed(breed1, breed2):
    if breed1 == breed2 or breed2 == 0:
        return 'Pure breed'
    else:
        return 'Mixed'


def health(train):
    palette1 = {"Healthy": "#628AD0", "Minor injury": "#EE7DC2", "Serious injury": "#E8C547",
                "Not specified": "#48E865"}
    palette2 = {"Not sure": "#EE7DC2", "Yes": "#E8C547", "No": "#48E865"}

    order = ['Same day', '1-7 days', '8-30 days', '31-90 days', 'No adoption']

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    g1 = sns.countplot(x='AdoptionSpeed', hue="Health", data=train, palette=palette1, ax=axes[0, 0], order=order)
    for p in g1.patches:
        x = p.get_bbox().get_points()[:, 0]
        y = p.get_bbox().get_points()[1, 1]
        g1.annotate('{:.2g}%'.format(100. * y / len(train)), (x.mean(), y), ha='center', va='bottom')

    g2 = sns.countplot(x='AdoptionSpeed', hue="Vaccinated", data=train, palette=palette2, ax=axes[0, 1], order=order)
    for p in g2.patches:
        x = p.get_bbox().get_points()[:, 0]
        y = p.get_bbox().get_points()[1, 1]
        g2.annotate('{:.2g}%'.format(100. * y / len(train)), (x.mean(), y), ha='center', va='bottom')

    g3 = sns.countplot(x='AdoptionSpeed', hue="Dewormed", data=train, palette=palette2, ax=axes[1, 0], order=order)
    for p in g3.patches:
        x = p.get_bbox().get_points()[:, 0]
        y = p.get_bbox().get_points()[1, 1]
        g3.annotate('{:.2g}%'.format(100. * y / len(train)), (x.mean(), y), ha='center', va='bottom')

    g4 = sns.countplot(x='AdoptionSpeed', hue="Sterilized", data=train, palette=palette2, ax=axes[1, 1], order=order)
    for p in g4.patches:
        x = p.get_bbox().get_points()[:, 0]
        y = p.get_bbox().get_points()[1, 1]
        g4.annotate('{:.2g}%'.format(100. * y / len(train)), (x.mean(), y), ha='center', va='bottom')

    plt.show()


def menu():
    while True:
        print("Select an option:")
        print("1 - Number of null values")
        print("2 - Animal type stats")
        print("3 - Breed stats")
        print("4 - Health and medical stats")
        print("X - Back")

        choice = input("Enter option number: ")

        train = load_and_map_data();
        palette = {"Dog": "#628AD0", "Cat": "#EE7DC2"}
        
        if choice == "1":
            null_values(train)
        elif choice == "2":
            animal_types(train, palette)
        elif choice == "3":
            breeds_stats(train, palette)
        elif choice == "4":
            health(train)
        elif choice == "x" or choice == "X":
            print("Goodbye!\n")
            break
        else:
            print("Invalid choice. Please select an option from menu.")

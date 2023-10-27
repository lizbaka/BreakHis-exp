import datasets
import pandas as pd

MY_FOLD_CSV = './dataset/BreaKHis_v1/mysplit.csv'

def dump_myfold_csv():
    data_list = []

    for magnification in [40, 100, 200, 400]:
        train_dataset_bin = {}
        dev_dataset_bin = {}
        test_dataset_bin = {}
        train_dataset_bin[magnification] = datasets.BreaKHis_generate('binary', 'train', magnification)
        dev_dataset_bin[magnification] = datasets.BreaKHis_generate('binary', 'dev', magnification)
        test_dataset_bin[magnification] = datasets.BreaKHis_generate('binary', 'test', magnification)
        for img in train_dataset_bin[magnification].img_list:
            data_list.append({'mag_grp':int(magnification), \
                            'grp':'train', \
                            'tumor_class':img.info['tumor_class'], \
                            'tumor_type':img.info['tumor_type'], \
                            'path':img.path.removeprefix('./dataset/BreaKHis_v1/')})
        for img in dev_dataset_bin[magnification].img_list:
            data_list.append({'mag_grp':int(magnification), \
                            'grp':'dev', \
                            'tumor_class':img.info['tumor_class'], \
                            'tumor_type':img.info['tumor_type'], \
                            'path':img.path.removeprefix('./dataset/BreaKHis_v1/')})
        for img in test_dataset_bin[magnification].img_list:
            data_list.append({'mag_grp':int(magnification), \
                            'grp':'test', \
                            'tumor_class':img.info['tumor_class'], \
                            'tumor_type':img.info['tumor_type'], \
                            'path':img.path.removeprefix('./dataset/BreaKHis_v1/')})

    df = pd.DataFrame(data_list)
    df.to_csv(MY_FOLD_CSV, index=False)


def statistic_myfold():
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(MY_FOLD_CSV)

    # Group the data by mag_grp and tumor_type, and count the number of occurrences
    counts = df.groupby(['mag_grp', 'tumor_type']).size().reset_index(name='count')

    # Save the counts to a new CSV file
    counts.to_csv('tumor_counts.csv', index=False)

    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv('tumor_counts.csv')

    # Pivot the data to create a table with mag_grp as rows and tumor_type as columns
    table = df.pivot(index='mag_grp', columns='tumor_type', values='count')

    # Fill any missing values with 0
    table = table.fillna(0)

    # dump the table to a new CSV file
    table.to_csv('tumor_table.csv')

if __name__ == '__main__':
    dump_myfold_csv()
    # statistic_myfold()
    pass
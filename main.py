import pandas as pd

from utils import read_data


if __name__ == '__main__':
    train_data, valid_data = read_data()
    train_df = pd.DataFrame(train_data, columns=['id', 'source', 'target', 'label'])




# -i https://pypi.tuna.tsinghua.edu.cn/simple
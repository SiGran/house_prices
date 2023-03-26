from sklearn.decomposition import PCA

def reduce_components(train, test, pca_option, n_components):
    """
    :param train: train data set
    :param test: test data set
    :param pca_option: 'no_pca', 'pca', 'kpca'
    :param n_components: number of components
    :return: train, test data set
    """
    x_train = train.drop(columns=['price'])
    if pca_option == 'pca':
        pca = PCA(n_components=n_components).fit(x_train)
        x_train = pca.transform(x_train)
        x_test = pca.transform(test.drop(columns=['price']))

    elif pca_option == 'no_pca':
        x_test = test.drop(columns=['price'])
    else:
        raise ValueError('pca_option not valid')
    
    return x_train, x_test

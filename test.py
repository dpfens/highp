if __name__ == '__main__':
    from highp import distance, dbscan, fuzzy
    from random import randint
    dist = distance.euclidean([1,2,3], [2,3,4])
    print(dist)
    clf = dbscan.NormalDBSCAN(5, 3, distance.euclidean)
    data = [[randint(0, 25) for j in range(3)] for i in range(10000)]
    clusters = clf.predict(data)
    for row in clusters:
        print(row)
    clf = fuzzy.FuzzyBorderDBSCAN(2.0, 20.0, 2, distance.euclidean)
    print('Fuzzy Clusters')
    clusters = clf.predict(data)
    for value, row in zip(data, clusters):
        print(value, dict(row))
    print('clustering complete')

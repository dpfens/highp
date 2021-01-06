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

    min_points = 2
    max_points = 5
    min_eps = 2.0
    max_eps = 20.0
    clf = fuzzy.CoreBorderDBSCAN(min_eps, min_points, max_points, distance.euclidean)
    clf = fuzzy.FuzzyBorderDBSCAN(min_eps, max_eps, min_points, distance.euclidean)
    print('Fuzzy Border Clusters')
    clusters = clf.predict(data)
    for value, row in zip(data, clusters):
        print(value, dict(row))

    clf = fuzzy.FuzzyDBSCAN(min_eps, max_eps, min_points, max_points, distance.euclidean
    print('clustering complete')

#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <vector>

#include "kdtree.hpp"

template <typename T>
KDNode<T>::KDNode() = default;

template <typename T>
KDNode<T>::KDNode(const std::vector<T> &pt, const size_t &idx_, const KDNodePtr &left_,
               const KDNodePtr &right_) {
    x = pt;
    index = idx_;
    left = left_;
    right = right_;
}

template <typename T>
KDNode<T>::KDNode(const typename std::pair< std::vector<T>, size_t> &pi, const KDNodePtr &left_,
               const KDNodePtr &right_) {
    x = pi.first;
    index = pi.second;
    left = left_;
    right = right_;
}

template <typename T>
KDNode<T>::~KDNode() = default;

template <typename T>
T KDNode<T>::coord(const size_t &idx) { return x.at(idx); }
template <typename T>
KDNode<T>::operator bool() { return (!x.empty()); }
template <typename T>
KDNode<T>::operator std::vector<T>() { return x; }
template <typename T>
KDNode<T>::operator size_t() { return index; }
template <typename T>
KDNode<T>::operator std::pair< std::vector< double >, size_t >() { return std::pair< std::vector< double >, size_t >(x, index); }

template <typename T>
KDNodePtr<T> NewKDNodePtr() {
    KDNodePtr<T> mynode = std::make_shared< KDNode<T> >();
    return mynode;
}

template <typename T>
inline double dist2(const std::vector<T> &a, const std::vector<T> &b) {
    double distc = 0;
    for (size_t i = 0; i < a.size(); i++) {
        double di = a.at(i) - b.at(i);
        distc += di * di;
    }
    return distc;
}

template <typename T>
inline double dist2(const KDNodePtr<T> &a, const KDNodePtr<T> &b) {
    return dist2(a->x, b->x);
}

template <typename T>
comparer<T>::comparer(size_t idx_) : idx{idx_} {};

template <typename T>
inline bool comparer<T>::compare_idx(const typename std::pair< std::vector<T>, size_t> &a,
                                  const typename std::pair< std::vector<T>, size_t> &b
) {
    return (a.first.at(idx) < b.first.at(idx));
}

template <typename T>
inline void sort_on_idx(const typename std::vector< std::pair< std::vector<T>, size_t> >::iterator &begin,
                        const typename std::vector< std::pair< std::vector<T>, size_t> >::iterator &end,
                        size_t idx) {
    comparer<T> comp(idx);
    comp.idx = idx;

    using std::placeholders::_1;
    using std::placeholders::_2;

    std::sort(begin, end, std::bind(&comparer<T>::compare_idx, comp, _1, _2));
}

template <typename T>
KDNodePtr<T> KDTree<T>::make_tree(const typename std::vector< std::pair< std::vector<T>, size_t> >::iterator &begin,
                            const typename std::vector< std::pair< std::vector<T>, size_t> >::iterator &end,
                            const size_t &length,
                            const size_t &level
) {
    if (begin == end) {
        return NewKDNodePtr<T>();
    }

    size_t dim = begin->first.size();

    if (length > 1) {
        sort_on_idx<T>(begin, end, level);
    }

    auto middle = begin + (length / 2);

    auto l_begin = begin;
    auto l_end = middle;
    auto r_begin = middle + 1;
    auto r_end = end;

    size_t l_len = length / 2;
    size_t r_len = length - l_len - 1;

    KDNodePtr<T> left;
    if (l_len > 0 && dim > 0) {
        left = make_tree(l_begin, l_end, l_len, (level + 1) % dim);
    } else {
        left = leaf;
    }
    KDNodePtr<T> right;
    if (r_len > 0 && dim > 0) {
        right = make_tree(r_begin, r_end, r_len, (level + 1) % dim);
    } else {
        right = leaf;
    }

    // KDNode result = KDNode();
    return std::make_shared< KDNode<T> >(*middle, left, right);
}

template <typename T>
KDTree<T>::KDTree(std::vector< std::vector<T> > point_array) {
    leaf = std::make_shared< KDNode<T> >();
    // iterators
    std::vector< std::pair< std::vector<T>, size_t> > arr;
    for (size_t i = 0; i < point_array.size(); i++) {
        arr.push_back(std::pair< std::vector<T>, size_t>(point_array.at(i), i));
    }

    auto begin = arr.begin();
    auto end = arr.end();

    size_t length = arr.size();
    size_t level = 0;  // starting

    root = KDTree::make_tree(begin, end, length, level);
}

template <typename T>
KDNodePtr<T> KDTree<T>::nearest_(
    const KDNodePtr<T> &branch,
    const std::vector<T> &pt,
    const size_t &level,
    const KDNodePtr<T> &best,
    const double &best_dist
) {
    double d, dx, dx2;

    if (!bool(*branch)) {
        return NewKDNodePtr<T>();  // basically, null
    }

    std::vector<T> branch_pt(*branch);
    size_t dim = branch_pt.size();

    d = dist2(branch_pt, pt);
    dx = branch_pt.at(level) - pt.at(level);
    dx2 = dx * dx;

    KDNodePtr<T> best_l = best;
    double best_dist_l = best_dist;

    if (d < best_dist) {
        best_dist_l = d;
        best_l = branch;
    }

    size_t next_lv = (level + 1) % dim;
    KDNodePtr<T> section;
    KDNodePtr<T> other;

    // select which branch makes sense to check
    if (dx > 0) {
        section = branch->left;
        other = branch->right;
    } else {
        section = branch->right;
        other = branch->left;
    }

    // keep nearest neighbor from further down the tree
    KDNodePtr<T> further = nearest_(section, pt, next_lv, best_l, best_dist_l);
    if (!further->x.empty()) {
        double dl = dist2(further->x, pt);
        if (dl < best_dist_l) {
            best_dist_l = dl;
            best_l = further;
        }
    }
    // only check the other branch if it makes sense to do so
    if (dx2 < best_dist_l) {
        further = nearest_(other, pt, next_lv, best_l, best_dist_l);
        if (!further->x.empty()) {
            double dl = dist2(further->x, pt);
            if (dl < best_dist_l) {
                best_dist_l = dl;
                best_l = further;
            }
        }
    }

    return best_l;
};

// default caller
template <typename T>
KDNodePtr<T> KDTree<T>::nearest_(const std::vector<T> &pt) {
    size_t level = 0;
    // KDNodePtr best = branch;
    double branch_dist = dist2(std::vector<T>(*root), pt);
    return nearest_(root,          // beginning of tree
                    pt,            // point we are querying
                    level,         // start from level 0
                    root,          // best is the root
                    branch_dist);  // best_dist = branch_dist
};

template <typename T>
std::vector<T> KDTree<T>::nearest_point(const std::vector<T> &pt) {
    return std::vector<T>(*nearest_(pt));
};

template <typename T>
size_t KDTree<T>::nearest_index(const std::vector<T> &pt) {
    return size_t(*nearest_(pt));
};

template <typename T>
typename std::pair< std::vector<T>, size_t> KDTree<T>::nearest_pointIndex(const std::vector<T> &pt) {
    KDNodePtr<T> Nearest = nearest_(pt);
    return pointIndex(std::vector<T>(*Nearest), size_t(*Nearest));
}

template <typename T>
std::vector< std::pair< std::vector<T>, size_t> > KDTree<T>::neighborhood_(
    const KDNodePtr<T> &branch,
    const std::vector<T> &pt,
    const double &rad,
    const size_t &level
) {
    double d, dx, dx2;

    if (!bool(*branch)) {
        // branch has no point, means it is a leaf,
        // no points to add
        return std::vector< std::pair< std::vector<T>, size_t> >();
    }

    size_t dim = pt.size();

    double r2 = rad * rad;

    d = dist2(std::vector<T>(*branch), pt);
    dx = std::vector<T>(*branch).at(level) - pt.at(level);
    dx2 = dx * dx;

    std::vector< std::pair< std::vector<T>, size_t> > nbh, nbh_s, nbh_o;
    if (d <= r2) {
        nbh.push_back(pointIndex(*branch));
    }

    KDNodePtr<T> section;
    KDNodePtr<T> other;
    if (dx > 0) {
        section = branch->left;
        other = branch->right;
    } else {
        section = branch->right;
        other = branch->left;
    }

    nbh_s = neighborhood_(section, pt, rad, (level + 1) % dim);
    nbh.insert(nbh.end(), nbh_s.begin(), nbh_s.end());
    if (dx2 < r2) {
        nbh_o = neighborhood_(other, pt, rad, (level + 1) % dim);
        nbh.insert(nbh.end(), nbh_o.begin(), nbh_o.end());
    }

    return nbh;
};

template <typename T>
std::vector< std::pair< std::vector<T>, size_t> > KDTree<T>::neighborhood(
    const std::vector<T> &pt,
    const double &rad) {
    size_t level = 0;
    return neighborhood_(root, pt, rad, level);
}

template <typename T>
std::vector< std::vector<T> > KDTree<T>::neighborhood_points(
    const std::vector<T> &pt,
    const double &rad) {
    size_t level = 0;
    std::vector< std::pair< std::vector<T>, size_t> > nbh = neighborhood_(root, pt, rad, level);
    std::vector< std::vector<T> > nbhp;
    nbhp.resize(nbh.size());
    std::transform(nbh.begin(), nbh.end(), nbhp.begin(),
                   [](typename std::pair< std::vector<T>, size_t> x) { return x.first; });
    return nbhp;
}

template <typename T>
std::vector<size_t> KDTree<T>::neighborhood_indices(
    const std::vector<T> &pt,
    const double &rad) {
    size_t level = 0;
    std::vector< std::pair< std::vector<T>, size_t> > nbh = neighborhood_(root, pt, rad, level);
    std::vector<size_t> nbhi;
    nbhi.resize(nbh.size());
    std::transform(nbh.begin(), nbh.end(), nbhi.begin(),
                   [](typename std::pair< std::vector<T>, size_t> x) { return x.second; });
    return nbhi;
}

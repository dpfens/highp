#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

using indexArr = std::vector<size_t>;

template <typename T>
class KDNode {
   public:
    using KDNodePtr = std::shared_ptr< KDNode<T> >;
    size_t index;
    std::vector<T> x;
    KDNodePtr left;
    KDNodePtr right;

    // initializer
    KDNode();
    KDNode(const std::vector<T> &, const size_t &, const KDNodePtr &,
           const KDNodePtr &);
    KDNode(const std::pair< std::vector<T>, size_t> &, const KDNodePtr &, const KDNodePtr &);
    ~KDNode();

    // getter
    T coord(const size_t &);

    // conversions
    explicit operator bool();
    explicit operator std::vector<T>();
    explicit operator size_t();
    explicit operator typename std::pair< std::vector< double >, size_t >();
};

template <typename T>
using KDNodePtr = std::shared_ptr< KDNode<T> >;

template <typename T>
inline double dist(const std::vector<T> &, const std::vector<T> &);
template <typename T>
inline double dist(const KDNodePtr<T> &, const KDNodePtr<T> &);

// Need for sorting
template <typename T>
class comparer {
   public:
    size_t idx;
    explicit comparer(size_t idx_);
    inline bool compare_idx(
        const std::pair< std::vector<T>, size_t > &,  //
        const std::pair< std::vector<T>, size_t > &   //
    );
};

template <typename T>
inline void sort_on_idx(const typename std::vector< std::pair< std::vector<T>, size_t> >::iterator &,  //
                        const typename std::vector< std::pair< std::vector<T>, size_t> >::iterator &,  //
                        size_t idx);

template <typename T>
class KDTree {
    KDNodePtr<T> root;
    KDNodePtr<T> leaf;

    KDNodePtr<T> make_tree(const typename std::vector< std::pair< std::vector<T>, size_t> >::iterator &begin,  //
                        const typename std::vector< std::pair< std::vector<T>, size_t> >::iterator &end,    //
                        const size_t &length,                  //
                        const size_t &level                    //
    );

   public:
    KDTree() = default;
    explicit KDTree(std::vector<std::vector<T> > point_array);

   private:
    KDNodePtr<T> nearest_(           //
        const KDNodePtr<T> &branch,  //
        const std::vector<T> &pt,        //
        const size_t &level,      //
        const KDNodePtr<T> &best,    //
        const double &best_dist   //
    );

    // default caller
    KDNodePtr<T> nearest_(const std::vector<T> &pt);

   public:
    std::vector<T> nearest_point(const std::vector<T> &pt);
    size_t nearest_index(const std::vector<T> &pt);
    std::pair< std::vector<T>, size_t> nearest_pointIndex(const std::vector<T> &pt);

   private:
    std::vector< std::pair< std::vector<T>, size_t> > neighborhood_(
        const KDNodePtr<T> &branch, const std::vector<T> &pt,
        const double &rad, const size_t &level);

   public:
    std::vector< std::pair< std::vector<T>, size_t> > neighborhood(
        const std::vector<T> &pt, const double &rad);

    std::vector<std::vector<T> > neighborhood_points(
        const std::vector<T> &pt, const double &rad);

    std::vector<size_t> neighborhood_indices(
        const std::vector<T> &pt, const double &rad);
};

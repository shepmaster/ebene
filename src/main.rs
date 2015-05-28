use std::u64;
use std::cmp::{min,max};

pub type Position = u64;
const EPSILON: Position = 1;
const NEGATIVE_INFINITY: Position = u64::MIN;
const POSITIVE_INFINITY: Position = u64::MAX;

pub type Extent = (Position, Position);
const START_EXTENT: Extent = (NEGATIVE_INFINITY, NEGATIVE_INFINITY);
const END_EXTENT: Extent = (POSITIVE_INFINITY, POSITIVE_INFINITY);

/// The basic query algebra from the [Clarke *et al.* paper][paper]
///
/// [paper]: http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.330.8436&rank=1
#[allow(unused_variables)]
pub trait Algebra {
    /// The first extent starting at or after the position k.
    fn tau(&self, k: Position) -> Extent { unimplemented!() }

    /// The last extent ending at or before the position k.
    ///
    /// This is akin to running `tau` from the other end of the number
    /// line. We are interested in the *first* number we arrive at
    /// (the end of the extent). We take the *first* extent that
    /// passes the criteria (the last extent in order).
    fn tau_prime(&self, k: Position) -> Extent { unimplemented!() }

    /// The first extent ending at or after the position k.
    fn rho(&self, k: Position) -> Extent { unimplemented!() }

    /// The last extent starting at or before the position k.
    ///
    /// This is akin to running `rho` from the other end of the number
    /// line. We are interested in the *second* number we arrive at
    /// (the start of the extent). We take the *first* extent that
    /// passes the criteria (the last extent in order).
    fn rho_prime(&self, k: Position) -> Extent { unimplemented!() }

    fn iter_tau(self) -> IterTau<Self>
        where Self: Sized
    {
        IterTau { list: self, k: NEGATIVE_INFINITY }
    }

    fn iter_rho(self) -> IterRho<Self>
        where Self: Sized
    {
        IterRho { list: self, k: NEGATIVE_INFINITY }
    }
}

#[derive(Debug,Copy,Clone)]
pub struct IterTau<T> {
    list: T,
    k: Position,
}

impl<T> Iterator for IterTau<T>
    where T: Algebra,
{
    type Item = Extent;
    fn next(&mut self) -> Option<Extent> {
        let (p, q) = self.list.tau(self.k);
        if p == POSITIVE_INFINITY { return None }

        self.k = p + EPSILON;
        Some((p, q))
    }
}

#[derive(Debug,Copy,Clone)]
pub struct IterRho<T> {
    list: T,
    k: Position,
}

impl<T> Iterator for IterRho<T>
    where T: Algebra,
{
    type Item = Extent;
    fn next(&mut self) -> Option<Extent> {
        let (p, q) = self.list.rho(self.k);
        if q == POSITIVE_INFINITY { return None }

        self.k = q + EPSILON;
        Some((p, q))
    }
}

// TODO: Investigate `get_unchecked` as we know the idx is valid.
impl<'a> Algebra for &'a [Extent] {
    fn tau(&self, k: Position) -> Extent {
        match self.binary_search_by(|ex| ex.0.cmp(&k)) {
            Ok(idx) => self[idx],
            Err(idx) if idx != self.len() => self[idx],
            Err(..) => END_EXTENT,
        }
    }

    // TODO: test
    fn tau_prime(&self, k: Position) -> Extent {
        match self.binary_search_by(|ex| ex.1.cmp(&k)) {
            Ok(idx) => self[idx],
            Err(idx) if idx != 0 => self[idx - 1],
            Err(..) => START_EXTENT,
        }
    }

    fn rho(&self, k: Position) -> Extent {
        match self.binary_search_by(|ex| ex.1.cmp(&k)) {
            Ok(idx) => self[idx],
            Err(idx) if idx != self.len() => self[idx],
            Err(..) => END_EXTENT,
        }
    }

    // TODO: test
    fn rho_prime(&self, k: Position) -> Extent {
        match self.binary_search_by(|ex| ex.0.cmp(&k)) {
            Ok(idx) => self[idx],
            Err(idx) if idx != 0 => self[idx - 1],
            Err(..) => START_EXTENT,
        }

    }
}

/// Returns extents from the first list that are contained in extents
/// from the second list.
///
/// Akin to finding needles in haystacks.
#[derive(Debug,Copy,Clone)]
pub struct ContainedIn<A, B>
    where A: Algebra,
          B: Algebra,
{
    a: A,
    b: B,
}

impl<A, B> Algebra for ContainedIn<A, B>
    where A: Algebra,
          B: Algebra,
{
    fn tau(&self, k: Position) -> Extent {
        let mut k = k;

        loop {
            let (p0, q0) = self.a.tau(k);
            let (p1, _)  = self.b.rho(q0);

            if p1 <= p0 {
                return (p0, q0);
            } else {
                // iteration instead of recursion
                k = p1;
            }
        }
    }

    // TODO: test
    fn rho(&self, k: Position) -> Extent {
        let (p, _) = self.a.rho(k);
        self.tau(p)
    }
}

/// Returns extents from the first list that contain extents from the
/// second list.
///
/// Akin to finding haystacks with needles in them.
#[derive(Debug,Copy,Clone)]
pub struct Containing<A, B>
    where A: Algebra,
          B: Algebra,
{
    a: A,
    b: B,
}

impl<A, B> Algebra for Containing<A, B>
    where A: Algebra,
          B: Algebra,
{
    fn tau(&self, k: Position) -> Extent {
        let (_, q) = self.a.tau(k);
        self.rho(q)
    }

    fn rho(&self, k: Position) -> Extent {
        let mut k = k;

        loop {
            let (p0, q0) = self.a.rho(k);
            let (_,  q1) = self.b.tau(p0);

            if q1 <= q0 {
                return (p0, q0);
            } else {
                // iteration instead of recursion
                k = q1;
            }
        }
    }
}

#[derive(Debug,Copy,Clone)]
pub struct NotContainedIn<A, B>
    where A: Algebra,
          B: Algebra,
{
    a: A,
    b: B,
}

impl<A, B> Algebra for NotContainedIn<A, B>
    where A: Algebra,
          B: Algebra,
{
    // TODO: test
    fn tau(&self, k: Position) -> Extent {
        let (p0, q0) = self.a.tau(k);
        let (p1, q1) = self.b.rho(q0);

        if p1 > p0 {
            (p0, q0)
        } else {
            // TODO: prevent recursion?
            self.rho(q1 + EPSILON)
        }
    }

    // TODO: test
    fn rho(&self, k: Position) -> Extent {
        let (p, _) = self.a.rho(k);
        self.tau(p)
    }
}

#[derive(Debug,Copy,Clone)]
pub struct NotContaining<A, B>
    where A: Algebra,
          B: Algebra,
{
    a: A,
    b: B,
}

impl<A, B> Algebra for NotContaining<A, B>
    where A: Algebra,
          B: Algebra,
{
    // TODO: test
    fn tau(&self, k: Position) -> Extent {
        let (_, q) = self.a.tau(k);
        self.rho(q)
    }

    // TODO: test
    fn rho(&self, k: Position) -> Extent {
        let (p0, q0) = self.a.rho(k);
        let (p1, q1) = self.b.tau(p0);

        if q1 > q0 {
            (p0, q0)
        } else {
            // TODO: prevent recursion?
            self.tau(p1 + EPSILON)
        }
    }
}

#[derive(Debug,Copy,Clone)]
pub struct BothOf<A, B>
    where A: Algebra,
          B: Algebra,
{
    a: A,
    b: B,
}

impl<A, B> Algebra for BothOf<A, B>
    where A: Algebra,
          B: Algebra,
{
    // TODO: test
    fn tau(&self, k: Position) -> Extent {
        // Find the farthest end of the next extents
        let (_,  q0) = self.a.tau(k);
        let (_,  q1) = self.b.tau(k);
        let max_q01  = max(q0, q1);

        // This line does not match the paper
        if max_q01 == POSITIVE_INFINITY { return END_EXTENT }

        // Find the extents prior to that point
        let (p2, q2) = self.a.tau_prime(max_q01);
        let (p3, q3) = self.b.tau_prime(max_q01);

        // Create a new extent that encompasses both preceeding extents
        (min(p2, p3), max(q2, q3))
    }

    // TODO: test
    fn rho(&self, k: Position) -> Extent {
        let (p, _) = self.tau_prime(k - EPSILON);
        self.tau(p + EPSILON)
    }
}

#[derive(Debug,Copy,Clone)]
pub struct OneOf<A, B>
    where A: Algebra,
          B: Algebra,
{
    a: A,
    b: B,
}

impl<A, B> Algebra for OneOf<A, B>
    where A: Algebra,
          B: Algebra,
{
    // TODO: test
    fn tau(&self, k: Position) -> Extent {
        // Find the extents after the point
        let (p0, q0) = self.a.tau(k);
        let (p1, q1) = self.b.tau(k);

        // TODO: use Ordering

        // Take the one that ends first, using the smaller extent in
        // case of ties
        if q0 < q1 {
            (p0, q0)
        } else if q0 > q1 {
            (p1, q1)
        } else {
            (max(p0, p1), q0)
        }
    }

    // TODO: test
    fn rho(&self, k: Position) -> Extent {
        // Find the extents ending after the point
        let (p0, q0) = self.a.rho(k);
        let (p1, q1) = self.b.rho(k);

        // TODO: use Ordering

        // Take the one that ends first, using the smaller extent in
        // case of ties
        if q0 < q1 {
            (p0, q0)
        } else if q0 > q1 {
            (p1, q1)
        } else {
            (max(p0, p1), q0)
        }
    }
}

#[derive(Debug,Copy,Clone)]
pub struct FollowedBy<A, B>
    where A: Algebra,
          B: Algebra,
{
    a: A,
    b: B,
}

impl<A, B> Algebra for FollowedBy<A, B>
    where A: Algebra,
          B: Algebra,
{
    // TODO: test
    fn tau(&self, k: Position) -> Extent {
        // Find the end of next extent after the point
        let (_,  q0) = self.a.tau(k);
        // Find the extent after the end
        let (p1, q1) = self.b.tau(q0 + EPSILON);
        // Look backwards for the extent that ends before the start
        let (p2, _)  = self.a.tau_prime(p1 - EPSILON);

        (p2, q1)
    }

    // TODO: test
    fn rho(&self, k: Position) -> Extent {
        let (p, _) = self.tau_prime(k - EPSILON);
        self.tau(p + EPSILON)
    }
}

extern crate quickcheck;
use quickcheck::{quickcheck,Arbitrary};

#[derive(Debug,Clone,PartialEq)]
struct RandomExtentList(Vec<Extent>);

impl Arbitrary for RandomExtentList {
    fn arbitrary<G>(g: &mut G) -> Self
        where G: quickcheck::Gen
    {
        let mut extents: Vec<Extent> = Arbitrary::arbitrary(g);

        // Reorder the extents
        let mut last_start = NEGATIVE_INFINITY;
        for extent in &mut extents {
            // Make sure the end comes after the start
            if extent.0 > extent.1 {
                *extent = (extent.1, extent.0);
            }
            // make sure that subsequent extents come after previous extents
            extent.0 += last_start;
            extent.1 += last_start;
            // Don't let the next extent overlap with us
            last_start = extent.1 + EPSILON;
        }

        RandomExtentList(extents)
    }

    fn shrink(&self) -> Box<Iterator<Item=Self>> {
        Box::new(self.0.shrink().map(|v| RandomExtentList(v)))
    }
}

impl Algebra for RandomExtentList {
    fn tau(&self, k: Position)       -> Extent { (&self.0[..]).tau(k) }
    fn tau_prime(&self, k: Position) -> Extent { (&self.0[..]).tau_prime(k) }
    fn rho(&self, k: Position)       -> Extent { (&self.0[..]).rho(k) }
    fn rho_prime(&self, k: Position) -> Extent { (&self.0[..]).rho_prime(k) }
}

fn main() {
}

#[test]
fn extent_list_all_tau_matches_all_rho() {
    fn prop(extents: RandomExtentList) -> bool {
        // TODO: add a reference or equivalent, instead of clone
        let a: Vec<_> = extents.clone().iter_tau().collect();
        let b: Vec<_> = extents.iter_rho().collect();

        // TODO: add a iterator comparison, not vec
        a == b
    }

    quickcheck(prop as fn(RandomExtentList) -> bool);
}

#[test]
fn extent_list_tau_finds_extents_that_start_at_same_point() {
    let a = &[(1,1), (2,2)][..];
    assert_eq!(a.tau(1), (1,1));
    assert_eq!(a.tau(2), (2,2));
}

#[test]
fn extent_list_tau_finds_first_extent_starting_after_point() {
    let a = &[(3,4)][..];
    assert_eq!(a.tau(1), (3,4));
}

#[test]
fn extent_list_tau_returns_end_marker_if_no_match() {
    let a = &[(1,3)][..];
    assert_eq!(a.tau(2), END_EXTENT);
}

#[test]
fn extent_list_rho_finds_extents_that_end_at_same_point() {
    let a = &[(1,1), (2,2)][..];
    assert_eq!(a.rho(1), (1,1));
    assert_eq!(a.rho(2), (2,2));
}

#[test]
fn extent_list_rho_finds_first_extent_ending_after_point() {
    let a = &[(3,4)][..];
    assert_eq!(a.rho(1), (3,4));
}

#[test]
fn extent_list_rho_returns_end_marker_if_no_match() {
    let a = &[(1,3)][..];
    assert_eq!(a.rho(4), END_EXTENT);
}

// Containing::rho is implemented in terms of tau, we should rewrite
// tests to leverage this

#[test]
fn contained_in_all_tau_matches_all_rho() {
    fn prop(a: RandomExtentList, b: RandomExtentList) -> bool {
        let c = ContainedIn { a: a, b: b };

        // TODO: add a reference or equivalent, instead of clone
        let a: Vec<_> = c.clone().iter_tau().collect();
        let b: Vec<_> = c.iter_rho().collect();

        // TODO: add a iterator comparison, not vec
        a == b
    }

    quickcheck(prop as fn(RandomExtentList, RandomExtentList) -> bool);
}

#[test]
fn contained_in_needle_is_fully_within_haystack() {
    let a = &[(2,3)][..];
    let b = &[(1,4)][..];
    let c = ContainedIn { a: a, b: b };
    assert_eq!(c.tau(1), (2,3));
}

#[test]
fn contained_in_needle_end_matches_haystack_end() {
    let a = &[(2,4)][..];
    let b = &[(1,4)][..];
    let c = ContainedIn { a: a, b: b };
    assert_eq!(c.tau(1), (2,4));
}

#[test]
fn contained_in_needle_start_matches_haystack_start() {
    let a = &[(1,3)][..];
    let b = &[(1,4)][..];
    let c = ContainedIn { a: a, b: b };
    assert_eq!(c.tau(1), (1,3));
}

#[test]
fn contained_in_needle_and_haystack_exactly_match() {
    let a = &[(1,4)][..];
    let b = &[(1,4)][..];
    let c = ContainedIn { a: a, b: b };
    assert_eq!(c.tau(1), (1,4));
}

#[test]
fn contained_in_needle_starts_too_early() {
    let a = &[(1,3)][..];
    let b = &[(2,4)][..];
    let c = ContainedIn { a: a, b: b };
    assert_eq!(c.tau(1), END_EXTENT);
}

#[test]
fn contained_in_needle_ends_too_late() {
    let a = &[(2,5)][..];
    let b = &[(1,4)][..];
    let c = ContainedIn { a: a, b: b };
    assert_eq!(c.tau(1), END_EXTENT);
}

// Containing::tau is implemented in terms of rho, no need for
// separate tests

#[test]
fn containing_all_tau_matches_all_rho() {
    fn prop(a: RandomExtentList, b: RandomExtentList) -> bool {
        let c = Containing { a: a, b: b };

        // TODO: add a reference or equivalent, instead of clone
        let a: Vec<_> = c.clone().iter_tau().collect();
        let b: Vec<_> = c.iter_rho().collect();

        // TODO: add a iterator comparison, not vec
        a == b
    }

    quickcheck(prop as fn(RandomExtentList, RandomExtentList) -> bool);
}

#[test]
fn containing_haystack_fully_around_needle() {
    let a = &[(1,4)][..];
    let b = &[(2,3)][..];
    let c = Containing { a: a, b: b };
    assert_eq!(c.tau(1), (1,4));
}

#[test]
fn containing_haystack_end_matches_needle_end() {
    let a = &[(1,4)][..];
    let b = &[(2,4)][..];
    let c = Containing { a: a, b: b };
    assert_eq!(c.tau(1), (1,4));
}

#[test]
fn containing_haystack_start_matches_needle_start() {
    let a = &[(1,4)][..];
    let b = &[(1,3)][..];
    let c = Containing { a: a, b: b };
    assert_eq!(c.tau(1), (1,4));
}

#[test]
fn containing_haystack_and_needle_exactly_match() {
    let a = &[(1,4)][..];
    let b = &[(1,4)][..];
    let c = Containing { a: a, b: b };
    assert_eq!(c.tau(1), (1,4));
}

#[test]
fn containing_haystack_starts_too_late() {
    let a = &[(2,4)][..];
    let b = &[(1,3)][..];
    let c = Containing { a: a, b: b };
    assert_eq!(c.tau(1), END_EXTENT);
}

#[test]
fn containing_haystack_ends_too_early() {
    let a = &[(1,4)][..];
    let b = &[(2,5)][..];
    let c = Containing { a: a, b: b };
    assert_eq!(c.tau(1), END_EXTENT);
}

#[test]
fn both_of_lists_have_extents_starting_after_point() {
    let a = &[(1,2)][..];
    let b = &[(3,4)][..];
    let c = BothOf { a: a, b: b };
    assert_eq!(c.tau(1), (1,4));
}

#[test]
fn both_of_lists_do_not_have_extents_starting_after_point() {
    let a = &[(1,2)][..];
    let b = &[(3,4)][..];
    let c = BothOf { a: a, b: b };
    assert_eq!(c.tau(5), END_EXTENT);
}

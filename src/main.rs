use std::u64;
use std::cmp::{min,max};

pub type Position = u64;
const EPSILON: Position = 1;
const NEGATIVE_INFINITY: Position = u64::MIN;
const POSITIVE_INFINITY: Position = u64::MAX;

/// Infinite values stay infinite when you add non-infinite values to
/// them.
///
/// NEGATIVE_INFINITY +/- EPSILON -> NEGATIVE_INFINITY
/// POSITIVE_INFINITY +/- EPSILON -> POSITIVE_INFINITY
trait Epsilon {
    fn increment(self) -> Self;
    fn decrement(self) -> Self;
}

impl Epsilon for Position {
    fn increment(self) -> Self {
        match self {
            x if x == NEGATIVE_INFINITY => self,
            x if x == POSITIVE_INFINITY => self,
            _ => self + 1,
        }
    }

    fn decrement(self) -> Self {
        match self {
            x if x == NEGATIVE_INFINITY => self,
            x if x == POSITIVE_INFINITY => self,
            _ => self - 1,
        }
    }
}

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

impl<'a, A: ?Sized> Algebra for &'a A
    where A: Algebra
{
    fn tau(&self, k: Position)       -> Extent { (*self).tau(k) }
    fn tau_prime(&self, k: Position) -> Extent { (*self).tau_prime(k) }
    fn rho(&self, k: Position)       -> Extent { (*self).rho(k) }
    fn rho_prime(&self, k: Position) -> Extent { (*self).rho_prime(k) }
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

        debug_assert!(self.k < p.increment());
        self.k = p.increment();
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

        debug_assert!(self.k < q.increment());
        self.k = q.increment();
        Some((p, q))
    }
}

macro_rules! check_forwards {
    ($k:expr) => { if $k == POSITIVE_INFINITY { return END_EXTENT } };
}

macro_rules! check_backwards {
    ($k:expr) => { if $k == NEGATIVE_INFINITY { return START_EXTENT } };
}

// TODO: Investigate `get_unchecked` as we know the idx is valid.
impl Algebra for [Extent] {
    fn tau(&self, k: Position) -> Extent {
        check_forwards!(k);
        match self.binary_search_by(|ex| ex.0.cmp(&k)) {
            Ok(idx) => self[idx],
            Err(idx) if idx != self.len() => self[idx],
            Err(..) => END_EXTENT,
        }
    }

    // TODO: test
    fn tau_prime(&self, k: Position) -> Extent {
        check_backwards!(k);
        match self.binary_search_by(|ex| ex.1.cmp(&k)) {
            Ok(idx) => self[idx],
            Err(idx) if idx != 0 => self[idx - 1],
            Err(..) => START_EXTENT,
        }
    }

    fn rho(&self, k: Position) -> Extent {
        check_forwards!(k);
        match self.binary_search_by(|ex| ex.1.cmp(&k)) {
            Ok(idx) => self[idx],
            Err(idx) if idx != self.len() => self[idx],
            Err(..) => END_EXTENT,
        }
    }

    // TODO: test
    fn rho_prime(&self, k: Position) -> Extent {
        check_backwards!(k);
        match self.binary_search_by(|ex| ex.0.cmp(&k)) {
            Ok(idx) => self[idx],
            Err(idx) if idx != 0 => self[idx - 1],
            Err(..) => START_EXTENT,
        }

    }
}

/// Finds extents from the first list that are contained in extents
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
            check_forwards!(k);

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
        check_forwards!(k);

        let (p, _) = self.a.rho(k);
        self.tau(p)
    }
}

/// Finds extents from the first list that contain extents from the
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
        check_forwards!(k);
        let (_, q) = self.a.tau(k);
        self.rho(q)
    }

    fn rho(&self, k: Position) -> Extent {
        let mut k = k;

        loop {
            check_forwards!(k);

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
        check_forwards!(k);
        let (p0, q0) = self.a.tau(k);
        let (p1, q1) = self.b.rho(q0);

        if p1 > p0 {
            (p0, q0)
        } else {
            // TODO: prevent recursion?
            self.rho(q1.increment())
        }
    }

    // TODO: test
    fn rho(&self, k: Position) -> Extent {
        check_forwards!(k);

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
        check_forwards!(k);

        let (_, q) = self.a.tau(k);
        self.rho(q)
    }

    // TODO: test
    fn rho(&self, k: Position) -> Extent {
        check_forwards!(k);

        let (p0, q0) = self.a.rho(k);
        let (p1, q1) = self.b.tau(p0);

        if q1 > q0 {
            (p0, q0)
        } else {
            // TODO: prevent recursion?
            self.tau(p1.increment())
        }
    }
}

/// Creates extents that extents from both lists would be a subextent
/// of.
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
        check_forwards!(k);

        // Find the farthest end of the next extents
        let (_, q0) = self.a.tau(k);
        let (_, q1) = self.b.tau(k);
        let max_q01 = max(q0, q1);

        // This line does not match the paper
        check_forwards!(max_q01);

        // Find the extents prior to that point
        let (p2, q2) = self.a.tau_prime(max_q01);
        let (p3, q3) = self.b.tau_prime(max_q01);

        // Create a new extent that encompasses both preceeding extents
        (min(p2, p3), max(q2, q3))
    }

    // TODO: test
    fn tau_prime(&self, k: Position) -> Extent {
        check_backwards!(k);

        let (p0, _) = self.a.tau_prime(k);
        let (p1, _) = self.b.tau_prime(k);
        let min_p01 = min(p0, p1);

        check_backwards!(min_p01);

        let (p2, q2) = self.a.tau(min_p01);
        let (p3, q3) = self.b.tau(min_p01);

        (min(p2, p3), max(q2, q3))
    }

    // TODO: test
    fn rho(&self, k: Position) -> Extent {
        check_forwards!(k);

        let (p, _) = self.tau_prime(k.decrement());
        self.tau(p.increment())
    }
}

/// Creates extents that an extent from either list would be a
/// subextent of.
///
/// # Note
///
/// `OneOf::tau` and `OneOf::rho` do *not* produce the same
/// list. As an example:
///
/// ```
/// A: (1,1)
/// B: (1,2)
/// ```
///
/// The only extent that *starts* in either input list would be
/// (1,1). There are two extents that *end* in either list: (1,1) and
/// (1,2). A similar construction exists for
///
/// ```
/// A: (1,2)
/// B: (2,2)
/// ```
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
        check_forwards!(k);

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
        check_forwards!(k);

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
        check_forwards!(k);

        // Find the first extent in A at or after the point
        let (_, q0) = self.a.tau(k);

        // Find the first extent in B at or after the first extent
        let (p1, q1) = self.b.tau(q0.increment());
        check_forwards!(q1);

        // Find the closest extent in A that is before the extent from B
        let (p2, _) = self.a.tau_prime(p1.decrement());
        (p2, q1)
    }

    // TODO: test
    fn tau_prime(&self, k: Position) -> Extent {
        check_backwards!(k);

        let (p0, _) = self.b.tau_prime(k);

        let (p1, q1) = self.a.tau_prime(p0.decrement());
        check_backwards!(q1);

        let (_, q2) = self.b.tau(q1.increment());
        (p1, q2)
    }

    // TODO: test
    fn rho(&self, k: Position) -> Extent {
        check_forwards!(k);

        let (p, _) = self.tau_prime(k.decrement());
        self.tau(p.increment())
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
        // Start at the first valid position. We can't use the sticky
        // infinity method here.
        let mut last_start = NEGATIVE_INFINITY + EPSILON;
        for extent in &mut extents {
            // Make sure the end comes after the start
            if extent.0 > extent.1 {
                *extent = (extent.1, extent.0);
            }
            // make sure that subsequent extents come after previous extents
            extent.0 += last_start;
            extent.1 += last_start;
            // Don't let the next extent overlap with us
            last_start = extent.1.increment();
        }

        RandomExtentList(extents)
    }

    fn shrink(&self) -> Box<Iterator<Item=Self>> {
        // Should avoid shrinking to START_EXTENT
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

fn iter_eq<A, B, T, U>(a: A, b: B) -> bool
    where A: IntoIterator<Item=T>,
          B: IntoIterator<Item=U>,
          T: PartialEq<U>,
{
    let mut a = a.into_iter();
    let mut b = b.into_iter();

    loop {
        match (a.next(), b.next()) {
            (Some(ref a), Some(ref b)) if a == b => continue,
            (None, None) => return true,
            _ => return false,
        }
    }
}

#[test]
fn extent_list_all_tau_matches_all_rho() {
    fn prop(extents: RandomExtentList) -> bool {
        let a = (&extents).iter_tau();
        let b = (&extents).iter_rho();

        iter_eq(a, b)
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
        let c = ContainedIn { a: &a, b: &b };

        let a = c.iter_tau();
        let b = c.iter_rho();

        iter_eq(a, b)
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
        let c = Containing { a: &a, b: &b };

        let a = c.iter_tau();
        let b = c.iter_rho();

        iter_eq(a, b)
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
fn both_of_all_tau_matches_all_rho() {
    fn prop(a: RandomExtentList, b: RandomExtentList) -> bool {
        let c = BothOf { a: &a, b: &b };

        let a = c.iter_tau();
        let b = c.iter_rho();

        iter_eq(a, b)
    }

    quickcheck(prop as fn(RandomExtentList, RandomExtentList) -> bool);
}

#[test]
fn both_of_initial_rho_doesnt_crash() {
    let a = &[][..];
    let b = &[][..];
    let c = BothOf { a: a, b: b };
    assert_eq!(c.rho(NEGATIVE_INFINITY), END_EXTENT);
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

#[test]
fn followed_by_all_tau_matches_all_rho() {
    fn prop(a: RandomExtentList, b: RandomExtentList) -> bool {
        let c = FollowedBy { a: &a, b: &b };

        let a = c.iter_tau();
        let b = c.iter_rho();

        iter_eq(a, b)
    }

    quickcheck(prop as fn(RandomExtentList, RandomExtentList) -> bool);
}

#[test]
fn followed_by_initial_rho_doesnt_crash() {
    let a = &[][..];
    let b = &[][..];
    let c = FollowedBy { a: a, b: b };
    assert_eq!(c.rho(NEGATIVE_INFINITY), END_EXTENT);
}

#[test]
fn followed_by_tau_prime_only_result() {
    let a = &[(1, 2)][..];
    let b = &[(3, 4)][..];
    let c = FollowedBy { a: a, b: b };
    assert_eq!(c.tau_prime(4), (1,4));
}

#[test]
fn followed_by_tau_prime_after_last_result() {
    let a = &[(1, 2)][..];
    let b = &[(3, 4)][..];
    let c = FollowedBy { a: a, b: b };
    assert_eq!(c.tau_prime(3), START_EXTENT);
}

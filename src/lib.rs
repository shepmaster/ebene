use std::u64;
use std::cmp::{min,max};
use Position::*;

#[derive(Debug,Copy,Clone,PartialEq,Eq,PartialOrd,Ord)]
pub enum Position {
    NegativeInfinity,
    Valid(u64),
    PositiveInfinity,
}

impl From<u64> for Position {
    fn from(v: u64) -> Position {
        Valid(v)
    }
}

trait Epsilon {
    fn increment(self) -> Self;
    fn decrement(self) -> Self;
}

impl Epsilon for Position {
    fn increment(self) -> Self {
        match self {
            NegativeInfinity |
            PositiveInfinity => self,
            Valid(x) if x == u64::MAX => PositiveInfinity,
            Valid(x) => Valid(x + 1),
        }
    }

    fn decrement(self) -> Self {
        match self {
            NegativeInfinity |
            PositiveInfinity => self,
            Valid(x) if x == u64::MIN => NegativeInfinity,
            Valid(x) => Valid(x - 1),
        }
    }
}

pub type ValidExtent = (u64, u64);

#[derive(Debug,Copy,Clone,PartialEq)]
pub struct Extent(pub Position, pub Position);
const START_EXTENT: Extent = Extent(NegativeInfinity, NegativeInfinity);
const END_EXTENT: Extent = Extent(PositiveInfinity, PositiveInfinity);

impl Extent {
    fn unwrap(self) -> ValidExtent {
        match self {
            Extent(Valid(a), Valid(b)) => (a, b),
            _ => panic!("Extent was not valid: {:?}", self),
        }
    }
}

impl PartialEq<ValidExtent> for Extent {
    fn eq(&self, other: &ValidExtent) -> bool {
        match *self {
            Extent(Valid(a), Valid(b)) => (a, b) == *other,
            _ => false,
        }
    }
}

impl From<ValidExtent> for Extent {
    fn from(v: ValidExtent) -> Extent {
        Extent(v.0.into(), v.1.into())
    }
}

/// The basic query algebra from the [Clarke *et al.* paper][paper]
///
/// # Iterators
///
/// Iterators are provided that return the entire result set in the
/// positive direction (tau, rho) and the negative direction
/// (tau-prime,rho-prime). Both forward iterators and both backwards
/// iterators return the same extents, and the forwards and backwards
/// iterators return the same extents in reverse order from each
/// other. All 4 iterators are provided for completeness.
///
/// # tau-prime and rho-prime
///
/// The paper does not give a concrete example of how to construct the
/// `*_prime` functions, simply stating
///
/// > The access functions τ′ and ρ′ are the converses of τ and ρ.
///
/// Through trial and error, I've determined that there are 4 concrete
/// steps to transform between prime and non-prime implementations:
///
/// 1. Swap usages of {tau,rho} with {tau-prime,rho-prime}
/// 2. Swap the sign of epsilon
/// 3. Swap the usages of p and q
/// 4. Swap comparison operators
///
/// [paper]: http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.330.8436&rank=1
#[allow(unused_variables)]
pub trait Algebra {
    /// The first extent starting at or after the position k.
    fn tau(&self, k: Position) -> Extent;

    /// The last extent ending at or before the position k.
    ///
    /// This is akin to running `tau` from the other end of the number
    /// line. We are interested in the *first* number we arrive at
    /// (the end of the extent). We take the *first* extent that
    /// passes the criteria (the last extent in order).
    fn tau_prime(&self, k: Position) -> Extent;

    /// The first extent ending at or after the position k.
    fn rho(&self, k: Position) -> Extent;

    /// The last extent starting at or before the position k.
    ///
    /// This is akin to running `rho` from the other end of the number
    /// line. We are interested in the *second* number we arrive at
    /// (the start of the extent). We take the *first* extent that
    /// passes the criteria (the last extent in order).
    fn rho_prime(&self, k: Position) -> Extent;

    /// Find all extents in a forward direction using the tau primitive
    fn iter_tau(self) -> IterTau<Self>
        where Self: Sized
    {
        IterTau { list: self, k: NegativeInfinity }
    }

    /// Find all extents in a forward direction using the rho primitive
    fn iter_rho(self) -> IterRho<Self>
        where Self: Sized
    {
        IterRho { list: self, k: NegativeInfinity }
    }

    /// Find all extents in a backward direction using the tau-prime primitive
    fn iter_tau_prime(self) -> IterTauPrime<Self>
        where Self: Sized
    {
        IterTauPrime { list: self, k: PositiveInfinity }
    }

    /// Find all extents in a backward direction using the rho-prime primitive
    fn iter_rho_prime(self) -> IterRhoPrime<Self>
        where Self: Sized
    {
        IterRhoPrime { list: self, k: PositiveInfinity }
    }
}

impl<'a, A: ?Sized> Algebra for Box<A>
    where A: Algebra
{
    fn tau(&self, k: Position)       -> Extent { (**self).tau(k) }
    fn tau_prime(&self, k: Position) -> Extent { (**self).tau_prime(k) }
    fn rho(&self, k: Position)       -> Extent { (**self).rho(k) }
    fn rho_prime(&self, k: Position) -> Extent { (**self).rho_prime(k) }
}

impl<'a, A: ?Sized> Algebra for &'a A
    where A: Algebra
{
    fn tau(&self, k: Position)       -> Extent { (**self).tau(k) }
    fn tau_prime(&self, k: Position) -> Extent { (**self).tau_prime(k) }
    fn rho(&self, k: Position)       -> Extent { (**self).rho(k) }
    fn rho_prime(&self, k: Position) -> Extent { (**self).rho_prime(k) }
}

/// Iterates over the extent list in the forward direction using the
/// tau primitive
#[derive(Debug,Copy,Clone)]
pub struct IterTau<T> {
    list: T,
    k: Position,
}

impl<T> Iterator for IterTau<T>
    where T: Algebra,
{
    type Item = ValidExtent;
    fn next(&mut self) -> Option<ValidExtent> {
        let Extent(p, q) = self.list.tau(self.k);
        if p == PositiveInfinity { return None }

        debug_assert!(self.k < p.increment());
        self.k = p.increment();
        Some(Extent(p, q).unwrap())
    }
}

/// Iterates over the extent list in the forward direction using the
/// rho primitive
#[derive(Debug,Copy,Clone)]
pub struct IterRho<T> {
    list: T,
    k: Position,
}

impl<T> Iterator for IterRho<T>
    where T: Algebra,
{
    type Item = ValidExtent;
    fn next(&mut self) -> Option<ValidExtent> {
        let Extent(p, q) = self.list.rho(self.k);
        if q == PositiveInfinity { return None }

        debug_assert!(self.k < q.increment());
        self.k = q.increment();
        Some(Extent(p, q).unwrap())
    }
}

/// Iterates over the extent list in the backward direction using the
/// tau-prime primitive
#[derive(Debug,Copy,Clone)]
pub struct IterTauPrime<T> {
    list: T,
    k: Position,
}

impl<T> Iterator for IterTauPrime<T>
    where T: Algebra,
{
    type Item = ValidExtent;
    fn next(&mut self) -> Option<ValidExtent> {
        let Extent(p, q) = self.list.tau_prime(self.k);
        if q == NegativeInfinity { return None }

        debug_assert!(self.k > q.decrement());
        self.k = q.decrement();
        Some(Extent(p, q).unwrap())
    }
}

/// Iterates over the extent list in the backward direction using the
/// rho-prime primitive
#[derive(Debug,Copy,Clone)]
pub struct IterRhoPrime<T> {
    list: T,
    k: Position,
}

impl<T> Iterator for IterRhoPrime<T>
    where T: Algebra,
{
    type Item = ValidExtent;
    fn next(&mut self) -> Option<ValidExtent> {
        let Extent(p, q) = self.list.rho_prime(self.k);
        if p == NegativeInfinity { return None }

        debug_assert!(self.k > p.decrement());
        self.k = p.decrement();
        Some(Extent(p, q).unwrap())
    }
}

macro_rules! check_forwards {
    ($k:expr) => { if $k == PositiveInfinity { return END_EXTENT; } };
}

macro_rules! check_backwards {
    ($k:expr) => { if $k == NegativeInfinity { return START_EXTENT; } };
}

macro_rules! check_and_unwrap_forwards {
    ($k:expr) => {
        match $k {
            NegativeInfinity => u64::MIN,
            Valid(x) => x,
            PositiveInfinity => return END_EXTENT,
        }
    }
}

macro_rules! check_and_unwrap_backwards {
    ($k:expr) => {
        match $k {
            NegativeInfinity => return START_EXTENT,
            Valid(x) => x,
            PositiveInfinity => u64::MAX,
        }
    }
}

// TODO: Investigate `get_unchecked` as we know the idx is valid.
impl Algebra for [ValidExtent] {
    fn tau(&self, k: Position) -> Extent {
        let k = check_and_unwrap_forwards!(k);
        match self.binary_search_by(|ex| ex.0.cmp(&k)) {
            Ok(idx) => self[idx].into(),
            Err(idx) if idx != self.len() => self[idx].into(),
            Err(..) => END_EXTENT,
        }
    }

    // TODO: test
    fn tau_prime(&self, k: Position) -> Extent {
        let k = check_and_unwrap_backwards!(k);
        match self.binary_search_by(|ex| ex.1.cmp(&k)) {
            Ok(idx) => self[idx].into(),
            Err(idx) if idx != 0 => self[idx - 1].into(),
            Err(..) => START_EXTENT,
        }
    }

    fn rho(&self, k: Position) -> Extent {
        let k = check_and_unwrap_forwards!(k);
        match self.binary_search_by(|ex| ex.1.cmp(&k)) {
            Ok(idx) => self[idx].into(),
            Err(idx) if idx != self.len() => self[idx].into(),
            Err(..) => END_EXTENT,
        }
    }

    // TODO: test
    fn rho_prime(&self, k: Position) -> Extent {
        let k = check_and_unwrap_backwards!(k);
        match self.binary_search_by(|ex| ex.0.cmp(&k)) {
            Ok(idx) => self[idx].into(),
            Err(idx) if idx != 0 => self[idx - 1].into(),
            Err(..) => START_EXTENT,
        }
    }
}

/// Finds no extents
pub struct Empty;

impl Algebra for Empty {
    fn tau(&self, _: Position)       -> Extent { END_EXTENT }
    fn tau_prime(&self, _: Position) -> Extent { START_EXTENT }
    fn rho(&self, _: Position)       -> Extent { END_EXTENT }
    fn rho_prime(&self, _: Position) -> Extent { START_EXTENT }
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

impl<A, B> ContainedIn<A, B>
    where A: Algebra,
          B: Algebra,
{
    pub fn new(a: A, b: B) -> Self { ContainedIn { a: a, b: b } }
}

impl<A, B> Algebra for ContainedIn<A, B>
    where A: Algebra,
          B: Algebra,
{
    fn tau(&self, k: Position) -> Extent {
        let mut k = k;

        loop {
            check_forwards!(k);

            let Extent(p0, q0) = self.a.tau(k);
            let Extent(p1, _)  = self.b.rho(q0);

            if p1 <= p0 {
                return Extent(p0, q0);
            } else {
                // iteration instead of recursion
                k = p1;
            }
        }
    }

    fn tau_prime(&self, k: Position) -> Extent {
        let mut k = k;

        loop {
            check_backwards!(k);

            let Extent(p0, q0) = self.a.tau_prime(k);
            let Extent(_,  q1) = self.b.rho_prime(p0);

            if q1 >= q0 {
                return Extent(p0, q0);
            } else {
                // iteration instead of recursion
                k = q1;
            }
        }
    }

    fn rho(&self, k: Position) -> Extent {
        check_forwards!(k);

        let Extent(p, _) = self.a.rho(k);
        self.tau(p)
    }

    fn rho_prime(&self, k: Position) -> Extent {
        check_backwards!(k);

        let Extent(_, q) = self.a.rho_prime(k);
        self.tau_prime(q)
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

impl<A, B> Containing<A, B>
    where A: Algebra,
          B: Algebra,
{
    pub fn new(a: A, b: B) -> Self { Containing { a: a, b: b } }
}

impl<A, B> Algebra for Containing<A, B>
    where A: Algebra,
          B: Algebra,
{
    fn tau(&self, k: Position) -> Extent {
        check_forwards!(k);
        let Extent(_, q) = self.a.tau(k);
        self.rho(q)
    }

    fn tau_prime(&self, k: Position) -> Extent {
        check_backwards!(k);
        let Extent(p, _) = self.a.tau_prime(k);
        self.rho_prime(p)
    }

    fn rho(&self, k: Position) -> Extent {
        let mut k = k;

        loop {
            check_forwards!(k);

            let Extent(p0, q0) = self.a.rho(k);
            let Extent(_,  q1) = self.b.tau(p0);

            if q1 <= q0 {
                return Extent(p0, q0);
            } else {
                // iteration instead of recursion
                k = q1;
            }
        }
    }

    fn rho_prime(&self, k: Position) -> Extent {
        let mut k = k;

        loop {
            check_backwards!(k);

            let Extent(p0, q0) = self.a.rho_prime(k);
            let Extent(p1, _)  = self.b.tau_prime(q0);

            if p1 >= p0 {
                return Extent(p0, q0);
            } else {
                // iteration instead of recursion
                k = p1;
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

impl<A, B> NotContainedIn<A, B>
    where A: Algebra,
          B: Algebra,
{
    pub fn new(a: A, b: B) -> Self { NotContainedIn { a: a, b: b } }
}

impl<A, B> Algebra for NotContainedIn<A, B>
    where A: Algebra,
          B: Algebra,
{
    fn tau(&self, k: Position) -> Extent {
        check_forwards!(k);
        let Extent(p0, q0) = self.a.tau(k);
        let Extent(p1, q1) = self.b.rho(q0);

        if p1 > p0 {
            Extent(p0, q0)
        } else {
            // TODO: prevent recursion?
            self.rho(q1.increment())
        }
    }

    fn tau_prime(&self, k: Position) -> Extent {
        check_backwards!(k);
        let Extent(p0, q0) = self.a.tau_prime(k);
        let Extent(p1, q1) = self.b.rho_prime(p0);

        if q1 < q0 {
            Extent(p0, q0)
        } else {
            // TODO: prevent recursion?
            self.rho_prime(p1.decrement())
        }
    }

    fn rho(&self, k: Position) -> Extent {
        check_forwards!(k);

        let Extent(p, _) = self.a.rho(k);
        self.tau(p)
    }

    fn rho_prime(&self, k: Position) -> Extent {
        check_backwards!(k);

        let Extent(_, q) = self.a.rho_prime(k);
        self.tau_prime(q)
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

impl<A, B> NotContaining<A, B>
    where A: Algebra,
          B: Algebra,
{
    pub fn new(a: A, b: B) -> Self { NotContaining { a: a, b: b } }
}

impl<A, B> Algebra for NotContaining<A, B>
    where A: Algebra,
          B: Algebra,
{
    fn tau(&self, k: Position) -> Extent {
        check_forwards!(k);

        let Extent(_, q) = self.a.tau(k);
        self.rho(q)
    }

    fn tau_prime(&self, k: Position) -> Extent {
        check_backwards!(k);

        let Extent(p, _) = self.a.tau_prime(k);
        self.rho_prime(p)
    }

    fn rho(&self, k: Position) -> Extent {
        check_forwards!(k);

        let Extent(p0, q0) = self.a.rho(k);
        let Extent(p1, q1) = self.b.tau(p0);

        if q1 > q0 {
            Extent(p0, q0)
        } else {
            // TODO: prevent recursion?
            self.tau(p1.increment())
        }
    }

    fn rho_prime(&self, k: Position) -> Extent {
        check_backwards!(k);

        let Extent(p0, q0) = self.a.rho_prime(k);
        let Extent(p1, q1) = self.b.tau_prime(q0);

        if p1 < p0 {
            Extent(p0, q0)
        } else {
            // TODO: prevent recursion?
           self.tau_prime(q1.decrement())
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

impl<A, B> BothOf<A, B>
    where A: Algebra,
          B: Algebra,
{
    pub fn new(a: A, b: B) -> Self { BothOf { a: a, b: b } }
}

impl<A, B> Algebra for BothOf<A, B>
    where A: Algebra,
          B: Algebra,
{
    fn tau(&self, k: Position) -> Extent {
        check_forwards!(k);

        // Find the farthest end of the next extents
        let Extent(_, q0) = self.a.tau(k);
        let Extent(_, q1) = self.b.tau(k);
        let max_q01 = max(q0, q1);

        // This line does not match the paper
        check_forwards!(max_q01);

        // Find the extents prior to that point
        let Extent(p2, q2) = self.a.tau_prime(max_q01);
        let Extent(p3, q3) = self.b.tau_prime(max_q01);

        // Create a new extent that encompasses both preceeding extents
        Extent(min(p2, p3), max(q2, q3))
    }

    fn tau_prime(&self, k: Position) -> Extent {
        check_backwards!(k);

        let Extent(p0, _) = self.a.tau_prime(k);
        let Extent(p1, _) = self.b.tau_prime(k);
        let min_p01 = min(p0, p1);

        check_backwards!(min_p01);

        let Extent(p2, q2) = self.a.tau(min_p01);
        let Extent(p3, q3) = self.b.tau(min_p01);

        Extent(min(p2, p3), max(q2, q3))
    }

    fn rho(&self, k: Position) -> Extent {
        check_forwards!(k);

        let Extent(p, _) = self.tau_prime(k.decrement());
        self.tau(p.increment())
    }

    fn rho_prime(&self, k: Position) -> Extent {
        check_backwards!(k);

        let Extent(_, q) = self.tau(k.increment());
        self.tau_prime(q.decrement())
    }
}

/// Finds extents that an extent from either list would be a subextent
/// of.
///
/// # Errors in the paper
///
/// Using the implementation in the paper, `OneOf::tau` and
/// `OneOf::rho` do *not* produce the same list. As an example:
///
/// ```text
///          k
/// |--*==*--|--|
/// *==|==|==|==*
/// 1  2  3  4  5
/// ```
///
/// `tau` would be correct for k=[0,5], but `rho` fails at k=[4,5],
/// producing (1,5).
///
/// To work around this, we work backward using tau_prime and then
/// forward again with tau, until we find a valid extent.
#[derive(Debug,Copy,Clone)]
pub struct OneOf<A, B>
    where A: Algebra,
          B: Algebra,
{
    a: A,
    b: B,
}

impl<A, B> OneOf<A, B>
    where A: Algebra,
          B: Algebra,
{
    pub fn new(a: A, b: B) -> Self { OneOf { a: a, b: b } }
}

impl<A, B> Algebra for OneOf<A, B>
    where A: Algebra,
          B: Algebra,
{
    fn tau(&self, k: Position) -> Extent {
        check_forwards!(k);

        // Find the extents after the point
        let Extent(p0, q0) = self.a.tau(k);
        let Extent(p1, q1) = self.b.tau(k);

        // TODO: use Ordering

        // Take the one that ends first, using the smaller extent in
        // case of ties
        if q0 < q1 {
            Extent(p0, q0)
        } else if q0 > q1 {
            Extent(p1, q1)
        } else {
            Extent(max(p0, p1), q0)
        }
    }

    fn tau_prime(&self, k: Position) -> Extent {
        check_backwards!(k);

        // Find the extents after the point
        let Extent(p0, q0) = self.a.tau_prime(k);
        let Extent(p1, q1) = self.b.tau_prime(k);

        // TODO: use Ordering

        // Take the one that ends first, using the smaller extent in
        // case of ties
        if p0 > p1 {
            Extent(p0, q0)
        } else if p0 < p1 {
            Extent(p1, q1)
        } else {
            Extent(p0, min(q0, q1))
        }
    }

    fn rho(&self, k: Position) -> Extent {
        check_forwards!(k);

        let Extent(p, q) = self.tau_prime(k);
        if q.increment() > k { return Extent(p, q) }

        loop {
            let Extent(p, q) = self.tau(p.increment());
            if q >= k { return Extent(p, q) }
        }
    }

    fn rho_prime(&self, k: Position) -> Extent {
        check_backwards!(k);

        let Extent(p, q) = self.tau(k);
        if p.decrement() < k { return Extent(p, q) }

        loop {
            let Extent(p, q) = self.tau_prime(q.decrement());
            if p <= k { return Extent(p, q) }
        }
    }
}

/// Creates extents that start at an extent from the first argument
/// and end at an extent from the second argument.
///
/// # tau-prime and rho-prime
///
/// In addition to the generic rules for constructing the prime
/// variants, FollowedBy requires that the A and B children be
/// swapped. This ensures that the ordering constraints are adhered,
/// otherwise we would find extents from B followed by extents from A.
///
#[derive(Debug,Copy,Clone)]
pub struct FollowedBy<A, B>
    where A: Algebra,
          B: Algebra,
{
    a: A,
    b: B,
}

impl<A, B> FollowedBy<A, B>
    where A: Algebra,
          B: Algebra,
{
    pub fn new(a: A, b: B) -> Self { FollowedBy { a: a, b: b } }
}

impl<A, B> Algebra for FollowedBy<A, B>
    where A: Algebra,
          B: Algebra,
{
    fn tau(&self, k: Position) -> Extent {
        check_forwards!(k);

        // Find the first extent in A at or after the point
        let Extent(_, q0) = self.a.tau(k);

        // Find the first extent in B at or after the first extent
        let Extent(p1, q1) = self.b.tau(q0.increment());
        check_forwards!(q1);

        // Find the closest extent in A that is before the extent from B
        let Extent(p2, _) = self.a.tau_prime(p1.decrement());
        Extent(p2, q1)
    }

    fn tau_prime(&self, k: Position) -> Extent {
        check_backwards!(k);

        let Extent(p0, _) = self.b.tau_prime(k);

        let Extent(p1, q1) = self.a.tau_prime(p0.decrement());
        check_backwards!(p1);

        let Extent(_, q2) = self.b.tau(q1.increment());
        Extent(p1, q2)
    }

    fn rho(&self, k: Position) -> Extent {
        check_forwards!(k);

        let Extent(p, _) = self.tau_prime(k.decrement());
        self.tau(p.increment())
    }

    fn rho_prime(&self, k: Position) -> Extent {
        check_backwards!(k);

        let Extent(_, q) = self.tau(k.increment());
        self.tau_prime(q.decrement())
    }
}

#[cfg(test)]
mod test {
    extern crate quickcheck;
    extern crate rand;

    use std::fmt::Debug;
    use self::quickcheck::{quickcheck,Arbitrary};
    use self::rand::Rng;

    use super::*;
    use super::END_EXTENT;

    fn find_invalid_gc_list_pair(extents: &[ValidExtent]) -> Option<(ValidExtent, ValidExtent)> {
        extents
            .windows(2)
            .map(|window| (window[0], window[1]))
            .find(|&(a, b)| b.0 <= a.0 || b.1 <= a.1)
    }

    fn assert_valid_gc_list(extents: &[ValidExtent]) {
        if let Some((a, b)) = find_invalid_gc_list_pair(extents) {
            assert!(false, "{:?} and {:?} are invalid GC-list members", a, b)
        }
    }

    impl Arbitrary for Position {
        fn arbitrary<G>(g: &mut G) -> Self
            where G: quickcheck::Gen
        {
            match g.gen_range(0, 10) {
                0 => Position::NegativeInfinity,
                1 => Position::PositiveInfinity,
                _ => Position::Valid(Arbitrary::arbitrary(g)),
            }
        }
    }

    #[derive(Debug,Clone,PartialEq)]
    struct RandomExtentList(Vec<ValidExtent>);

    impl Arbitrary for RandomExtentList {
        fn arbitrary<G>(g: &mut G) -> Self
            where G: quickcheck::Gen
        {
            let mut extents = vec![];
            let mut last_extent = (0, 1);

            for _ in 0..g.size() {
                let start_offset = u64::arbitrary(g);
                let new_start = last_extent.0 + 1 + start_offset;
                let min_width = last_extent.1 - last_extent.0;
                let max_width = min_width + g.size() as u64;
                let width = g.gen_range(min_width, max_width);

                let extent = (new_start, new_start + width);
                extents.push(extent);
                last_extent = extent;
            }

            assert_valid_gc_list(&extents);
            RandomExtentList(extents)
        }

        fn shrink(&self) -> Box<Iterator<Item=Self>> {
            Box::new(RandomExtentListShrinker(self.0.clone()))
        }
    }

    /// A simplistic shrinking strategy that preserves the ordering
    /// guarantee of the extent list
    struct RandomExtentListShrinker(Vec<ValidExtent>);

    impl Iterator for RandomExtentListShrinker {
        type Item = RandomExtentList;

        fn next(&mut self) -> Option<RandomExtentList> {
            match self.0.pop() {
                Some(..) => Some(RandomExtentList(self.0.clone())),
                None => None,
            }
        }
    }

    impl Algebra for RandomExtentList {
        fn tau(&self, k: Position)       -> Extent { (&self.0[..]).tau(k) }
        fn tau_prime(&self, k: Position) -> Extent { (&self.0[..]).tau_prime(k) }
        fn rho(&self, k: Position)       -> Extent { (&self.0[..]).rho(k) }
        fn rho_prime(&self, k: Position) -> Extent { (&self.0[..]).rho_prime(k) }
    }

    fn all_extents<A>(a: A) -> Vec<ValidExtent>
        where A: Algebra
    {
        a.iter_tau().collect()
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

    fn any_k<A>(operator: A, k: Position) -> bool
        where A: Algebra + Copy
    {
        let from_zero = all_extents(operator);

        let via_tau = operator.tau(k) == from_zero.tau(k);
        let via_rho = operator.rho(k) == from_zero.rho(k);
        let via_tau_prime = operator.tau_prime(k) == from_zero.tau_prime(k);
        let via_rho_prime = operator.rho_prime(k) == from_zero.rho_prime(k);

        via_tau && via_rho && via_tau_prime && via_rho_prime
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
        assert_eq!(a.tau(1.into()), (1,1));
        assert_eq!(a.tau(2.into()), (2,2));
    }

    #[test]
    fn extent_list_tau_finds_first_extent_starting_after_point() {
        let a = &[(3,4)][..];
        assert_eq!(a.tau(1.into()), (3,4));
    }

    #[test]
    fn extent_list_tau_returns_end_marker_if_no_match() {
        let a = &[(1,3)][..];
        assert_eq!(a.tau(2.into()), END_EXTENT);
    }

    #[test]
    fn extent_list_rho_finds_extents_that_end_at_same_point() {
        let a = &[(1,1), (2,2)][..];
        assert_eq!(a.rho(1.into()), (1,1));
        assert_eq!(a.rho(2.into()), (2,2));
    }

    #[test]
    fn extent_list_rho_finds_first_extent_ending_after_point() {
        let a = &[(3,4)][..];
        assert_eq!(a.rho(1.into()), (3,4));
    }

    #[test]
    fn extent_list_rho_returns_end_marker_if_no_match() {
        let a = &[(1,3)][..];
        assert_eq!(a.rho(4.into()), END_EXTENT);
    }

    #[test]
    fn contained_in_all_tau_matches_all_rho() {
        fn prop(a: RandomExtentList, b: RandomExtentList) -> bool {
            let c = ContainedIn { a: &a, b: &b };
            iter_eq(c.iter_tau(), c.iter_rho())
        }

        quickcheck(prop as fn(RandomExtentList, RandomExtentList) -> bool);
    }

    #[test]
    fn contained_in_all_tau_prime_matches_all_rho_prime() {
        fn prop(a: RandomExtentList, b: RandomExtentList) -> bool {
            let c = ContainedIn { a: &a, b: &b };
            iter_eq(c.iter_tau_prime(), c.iter_rho_prime())
        }

        quickcheck(prop as fn(RandomExtentList, RandomExtentList) -> bool);
    }

    #[test]
    fn contained_in_any_k() {
        fn prop(a: RandomExtentList, b: RandomExtentList, k: Position) -> bool {
            any_k(ContainedIn { a: &a, b: &b }, k)
        }

        quickcheck(prop as fn(RandomExtentList, RandomExtentList, Position) -> bool);
    }

    #[test]
    fn contained_in_needle_is_fully_within_haystack() {
        let a = &[(2,3)][..];
        let b = &[(1,4)][..];
        let c = ContainedIn { a: a, b: b };
        assert_eq!(c.tau(1.into()), (2,3));
    }

    #[test]
    fn contained_in_needle_end_matches_haystack_end() {
        let a = &[(2,4)][..];
        let b = &[(1,4)][..];
        let c = ContainedIn { a: a, b: b };
        assert_eq!(c.tau(1.into()), (2,4));
    }

    #[test]
    fn contained_in_needle_start_matches_haystack_start() {
        let a = &[(1,3)][..];
        let b = &[(1,4)][..];
        let c = ContainedIn { a: a, b: b };
        assert_eq!(c.tau(1.into()), (1,3));
    }

    #[test]
    fn contained_in_needle_and_haystack_exactly_match() {
        let a = &[(1,4)][..];
        let b = &[(1,4)][..];
        let c = ContainedIn { a: a, b: b };
        assert_eq!(c.tau(1.into()), (1,4));
    }

    #[test]
    fn contained_in_needle_starts_too_early() {
        let a = &[(1,3)][..];
        let b = &[(2,4)][..];
        let c = ContainedIn { a: a, b: b };
        assert_eq!(c.tau(1.into()), END_EXTENT);
    }

    #[test]
    fn contained_in_needle_ends_too_late() {
        let a = &[(2,5)][..];
        let b = &[(1,4)][..];
        let c = ContainedIn { a: a, b: b };
        assert_eq!(c.tau(1.into()), END_EXTENT);
    }

    #[test]
    fn containing_all_tau_matches_all_rho() {
        fn prop(a: RandomExtentList, b: RandomExtentList) -> bool {
            let c = Containing { a: &a, b: &b };
            iter_eq(c.iter_tau(), c.iter_rho())
        }

        quickcheck(prop as fn(RandomExtentList, RandomExtentList) -> bool);
    }

    #[test]
    fn containing_all_tau_prime_matches_all_rho_prime() {
        fn prop(a: RandomExtentList, b: RandomExtentList) -> bool {
            let c = Containing { a: &a, b: &b };
            iter_eq(c.iter_tau_prime(), c.iter_rho_prime())
        }

        quickcheck(prop as fn(RandomExtentList, RandomExtentList) -> bool);
    }

    #[test]
    fn containing_any_k() {
        fn prop(a: RandomExtentList, b: RandomExtentList, k: Position) -> bool {
            any_k(Containing { a: &a, b: &b }, k)
        }

        quickcheck(prop as fn(RandomExtentList, RandomExtentList, Position) -> bool);
    }

    #[test]
    fn containing_haystack_fully_around_needle() {
        let a = &[(1,4)][..];
        let b = &[(2,3)][..];
        let c = Containing { a: a, b: b };
        assert_eq!(c.tau(1.into()), (1,4));
    }

    #[test]
    fn containing_haystack_end_matches_needle_end() {
        let a = &[(1,4)][..];
        let b = &[(2,4)][..];
        let c = Containing { a: a, b: b };
        assert_eq!(c.tau(1.into()), (1,4));
    }

    #[test]
    fn containing_haystack_start_matches_needle_start() {
        let a = &[(1,4)][..];
        let b = &[(1,3)][..];
        let c = Containing { a: a, b: b };
        assert_eq!(c.tau(1.into()), (1,4));
    }

    #[test]
    fn containing_haystack_and_needle_exactly_match() {
        let a = &[(1,4)][..];
        let b = &[(1,4)][..];
        let c = Containing { a: a, b: b };
        assert_eq!(c.tau(1.into()), (1,4));
    }

    #[test]
    fn containing_haystack_starts_too_late() {
        let a = &[(2,4)][..];
        let b = &[(1,3)][..];
        let c = Containing { a: a, b: b };
        assert_eq!(c.tau(1.into()), END_EXTENT);
    }

    #[test]
    fn containing_haystack_ends_too_early() {
        let a = &[(1,4)][..];
        let b = &[(2,5)][..];
        let c = Containing { a: a, b: b };
        assert_eq!(c.tau(1.into()), END_EXTENT);
    }

    #[test]
    fn not_contained_in_all_tau_matches_all_rho() {
        fn prop(a: RandomExtentList, b: RandomExtentList) -> bool {
            let c = NotContainedIn { a: &a, b: &b };
            iter_eq(c.iter_tau(), c.iter_rho())
        }

        quickcheck(prop as fn(RandomExtentList, RandomExtentList) -> bool);
    }

    #[test]
    fn not_contained_in_all_tau_prime_matches_all_rho_prime() {
        fn prop(a: RandomExtentList, b: RandomExtentList) -> bool {
            let c = NotContainedIn { a: &a, b: &b };
            iter_eq(c.iter_tau_prime(), c.iter_rho_prime())
        }

        quickcheck(prop as fn(RandomExtentList, RandomExtentList) -> bool);
    }

    #[test]
    fn not_contained_in_any_k() {
        fn prop(a: RandomExtentList, b: RandomExtentList, k: Position) -> bool {
            any_k(NotContainedIn { a: &a, b: &b }, k)
        }

        quickcheck(prop as fn(RandomExtentList, RandomExtentList, Position) -> bool);
    }

    #[test]
    fn not_contained_in_needle_is_fully_within_haystack() {
        let a = &[(2,3)][..];
        let b = &[(1,4)][..];
        let c = NotContainedIn { a: a, b: b };
        assert_eq!(c.tau(1.into()), END_EXTENT);
    }

    #[test]
    fn not_contained_in_needle_end_matches_haystack_end() {
        let a = &[(2,4)][..];
        let b = &[(1,4)][..];
        let c = NotContainedIn { a: a, b: b };
        assert_eq!(c.tau(1.into()), END_EXTENT);
    }

    #[test]
    fn not_contained_in_needle_start_matches_haystack_start() {
        let a = &[(1,3)][..];
        let b = &[(1,4)][..];
        let c = NotContainedIn { a: a, b: b };
        assert_eq!(c.tau(1.into()), END_EXTENT);
    }

    #[test]
    fn not_contained_in_needle_and_haystack_exactly_match() {
        let a = &[(1,4)][..];
        let b = &[(1,4)][..];
        let c = NotContainedIn { a: a, b: b };
        assert_eq!(c.tau(1.into()), END_EXTENT);
    }

    #[test]
    fn not_contained_in_needle_starts_too_early() {
        let a = &[(1,3)][..];
        let b = &[(2,4)][..];
        let c = NotContainedIn { a: a, b: b };
        assert_eq!(c.tau(1.into()), (1,3));
    }

    #[test]
    fn not_contained_in_needle_ends_too_late() {
        let a = &[(2,5)][..];
        let b = &[(1,4)][..];
        let c = NotContainedIn { a: a, b: b };
        assert_eq!(c.tau(1.into()), (2,5));
    }

    #[test]
    fn not_containing_all_tau_matches_all_rho() {
        fn prop(a: RandomExtentList, b: RandomExtentList) -> bool {
            let c = NotContaining { a: &a, b: &b };
            iter_eq(c.iter_tau(), c.iter_rho())
        }

        quickcheck(prop as fn(RandomExtentList, RandomExtentList) -> bool);
    }

    #[test]
    fn not_containing_all_tau_prime_matches_all_rho_prime() {
        fn prop(a: RandomExtentList, b: RandomExtentList) -> bool {
            let c = NotContaining { a: &a, b: &b };
            iter_eq(c.iter_tau_prime(), c.iter_rho_prime())
        }

        quickcheck(prop as fn(RandomExtentList, RandomExtentList) -> bool);
    }

    #[test]
    fn not_containing_any_k() {
        fn prop(a: RandomExtentList, b: RandomExtentList, k: Position) -> bool {
            any_k(NotContaining { a: &a, b: &b }, k)
        }

        quickcheck(prop as fn(RandomExtentList, RandomExtentList, Position) -> bool);
    }

    #[test]
    fn not_containing_haystack_fully_around_needle() {
        let a = &[(1,4)][..];
        let b = &[(2,3)][..];
        let c = NotContaining { a: a, b: b };
        assert_eq!(c.tau(1.into()), END_EXTENT);
    }

    #[test]
    fn not_containing_haystack_end_matches_needle_end() {
        let a = &[(1,4)][..];
        let b = &[(2,4)][..];
        let c = NotContaining { a: a, b: b };
        assert_eq!(c.tau(1.into()), END_EXTENT);
    }

    #[test]
    fn not_containing_haystack_start_matches_needle_start() {
        let a = &[(1,4)][..];
        let b = &[(1,3)][..];
        let c = NotContaining { a: a, b: b };
        assert_eq!(c.tau(1.into()), END_EXTENT);
    }

    #[test]
    fn not_containing_haystack_and_needle_exactly_match() {
        let a = &[(1,4)][..];
        let b = &[(1,4)][..];
        let c = NotContaining { a: a, b: b };
        assert_eq!(c.tau(1.into()), END_EXTENT);
    }

    #[test]
    fn not_containing_haystack_starts_too_late() {
        let a = &[(2,4)][..];
        let b = &[(1,3)][..];
        let c = NotContaining { a: a, b: b };
        assert_eq!(c.tau(1.into()), (2,4));
    }

    #[test]
    fn not_containing_haystack_ends_too_early() {
        let a = &[(1,4)][..];
        let b = &[(2,5)][..];
        let c = NotContaining { a: a, b: b };
        assert_eq!(c.tau(1.into()), (1,4));
    }

    #[test]
    fn both_of_all_tau_matches_all_rho() {
        fn prop(a: RandomExtentList, b: RandomExtentList) -> bool {
            let c = BothOf { a: &a, b: &b };
            iter_eq(c.iter_tau(), c.iter_rho())
        }

        quickcheck(prop as fn(RandomExtentList, RandomExtentList) -> bool);
    }

    #[test]
    fn both_of_all_tau_prime_matches_all_rho_prime() {
        fn prop(a: RandomExtentList, b: RandomExtentList) -> bool {
            let c = BothOf { a: &a, b: &b };
            iter_eq(c.iter_tau_prime(), c.iter_rho_prime())
        }

        quickcheck(prop as fn(RandomExtentList, RandomExtentList) -> bool);
    }

    #[test]
    fn both_of_any_k() {
        fn prop(a: RandomExtentList, b: RandomExtentList, k: Position) -> bool {
            any_k(BothOf { a: &a, b: &b }, k)
        }

        quickcheck(prop as fn(RandomExtentList, RandomExtentList, Position) -> bool);
    }

    #[test]
    fn both_of_intersects_empty_lists() {
        let a = &[][..];
        let b = &[][..];
        let c = BothOf { a: &a, b: &b };

        assert_eq!(all_extents(c), []);
    }

    #[test]
    fn both_of_intersects_empty_list_and_nonempty_list() {
        let a = &[][..];
        let b = &[(1,2)][..];

        let c = BothOf { a: &a, b: &b };
        assert_eq!(all_extents(c), []);

        let c = BothOf { a: &b, b: &a };
        assert_eq!(all_extents(c), []);
    }

    #[test]
    fn both_of_intersects_nonempty_lists() {
        let a = &[(1,2)][..];
        let b = &[(3,4)][..];

        let c = BothOf { a: &a, b: &b };
        assert_eq!(all_extents(c), [(1,4)]);

        let c = BothOf { a: &b, b: &a };
        assert_eq!(all_extents(c), [(1,4)]);
    }

    #[test]
    fn both_of_intersects_overlapping_nonnested_lists() {
        let a = &[(1,3)][..];
        let b = &[(2,4)][..];

        let c = BothOf { a: &a, b: &b };
        assert_eq!(all_extents(c), [(1,4)]);

        let c = BothOf { a: &b, b: &a };
        assert_eq!(all_extents(c), [(1,4)]);
    }

    #[test]
    fn both_of_merges_overlapping_nested_lists() {
        let a = &[(1,4)][..];
        let b = &[(2,3)][..];

        let c = BothOf { a: &a, b: &b };
        assert_eq!(all_extents(c), [(1,4)]);

        let c = BothOf { a: &b, b: &a };
        assert_eq!(all_extents(c), [(1,4)]);
    }

    #[test]
    fn both_of_merges_overlapping_lists_nested_at_end() {
        let a = &[(1,4)][..];
        let b = &[(2,4)][..];

        let c = BothOf { a: &a, b: &b };
        assert_eq!(all_extents(c), [(1,4)]);

        let c = BothOf { a: &b, b: &a };
        assert_eq!(all_extents(c), [(1,4)]);
    }

    #[test]
    fn both_of_merges_overlapping_lists_nested_at_start() {
        let a = &[(1,4)][..];
        let b = &[(1,3)][..];

        let c = BothOf { a: &a, b: &b };
        assert_eq!(all_extents(c), [(1,4)]);

        let c = BothOf { a: &b, b: &a };
        assert_eq!(all_extents(c), [(1,4)]);
    }

    #[test]
    fn both_of_lists_have_extents_starting_after_point() {
        let a = &[(1,2)][..];
        let b = &[(3,4)][..];
        let c = BothOf { a: a, b: b };
        assert_eq!(c.tau(1.into()), (1,4));
    }

    #[test]
    fn both_of_lists_do_not_have_extents_starting_after_point() {
        let a = &[(1,2)][..];
        let b = &[(3,4)][..];
        let c = BothOf { a: a, b: b };
        assert_eq!(c.tau(5.into()), END_EXTENT);
    }

    #[test]
    fn one_of_all_tau_matches_all_rho() {
        fn prop(a: RandomExtentList, b: RandomExtentList) -> bool {
            let c = OneOf { a: &a, b: &b };
            iter_eq(c.iter_tau(), c.iter_rho())
        }

        quickcheck(prop as fn(RandomExtentList, RandomExtentList) -> bool);
    }

    #[test]
    fn one_of_any_k() {
        fn prop(a: RandomExtentList, b: RandomExtentList, k: Position) -> bool {
            any_k(OneOf { a: &a, b: &b }, k)
        }

        quickcheck(prop as fn(RandomExtentList, RandomExtentList, Position) -> bool);
    }

    #[test]
    fn one_of_merges_empty_lists() {
        let a = &[][..];
        let b = &[][..];
        let c = OneOf { a: &a, b: &b };

        assert_eq!(all_extents(c), []);
    }

    #[test]
    fn one_of_merges_empty_list_and_nonempty_list() {
        let a = &[][..];
        let b = &[(1,2)][..];

        let c = OneOf { a: &a, b: &b };
        assert_eq!(all_extents(c), [(1,2)]);

        let c = OneOf { a: &b, b: &a };
        assert_eq!(all_extents(c), [(1,2)]);
    }

    #[test]
    fn one_of_merges_nonempty_lists() {
        let a = &[(1,2)][..];
        let b = &[(3,4)][..];

        let c = OneOf { a: &a, b: &b };
        assert_eq!(all_extents(c), [(1,2), (3,4)]);

        let c = OneOf { a: &b, b: &a };
        assert_eq!(all_extents(c), [(1,2), (3,4)]);
    }

    #[test]
    fn one_of_merges_overlapping_nonnested_lists() {
        let a = &[(1,3)][..];
        let b = &[(2,4)][..];

        let c = OneOf { a: &a, b: &b };
        assert_eq!(all_extents(c), [(1,3), (2,4)]);

        let c = OneOf { a: &b, b: &a };
        assert_eq!(all_extents(c), [(1,3), (2,4)]);
    }

    #[test]
    fn one_of_merges_overlapping_nested_lists() {
        let a = &[(1,4)][..];
        let b = &[(2,3)][..];

        let c = OneOf { a: &a, b: &b };
        assert_eq!(all_extents(c), [(2,3)]);

        let c = OneOf { a: &b, b: &a };
        assert_eq!(all_extents(c), [(2,3)]);
    }

    #[test]
    fn one_of_merges_overlapping_lists_nested_at_end() {
        let a = &[(1,4)][..];
        let b = &[(2,4)][..];

        let c = OneOf { a: &a, b: &b };
        assert_eq!(all_extents(c), [(2,4)]);

        let c = OneOf { a: &b, b: &a };
        assert_eq!(all_extents(c), [(2,4)]);
    }

    #[test]
    fn one_of_merges_overlapping_lists_nested_at_start() {
        let a = &[(1,4)][..];
        let b = &[(1,3)][..];

        let c = OneOf { a: &a, b: &b };
        assert_eq!(all_extents(c), [(1,3)]);

        let c = OneOf { a: &b, b: &a };
        assert_eq!(all_extents(c), [(1,3)]);
    }

    // The paper has an incorrect implementation of OneOf::rho, so we take
    // the time to have some extra test cases exposed by quickcheck.

    #[test]
    fn one_of_rho_one_empty_list() {
        let a = &[][..];
        let b = &[(1, 2)][..];
        let c = OneOf { a: &a, b: &b };

        assert_eq!(c.rho(0.into()), (1,2));
        assert_eq!(c.rho(1.into()), (1,2));
        assert_eq!(c.rho(2.into()), (1,2));
        assert_eq!(c.rho(3.into()), END_EXTENT);
    }

    #[test]
    fn one_of_rho_nested_extents() {
        let a = &[(2, 3)][..];
        let b = &[(1, 5)][..];
        let c = OneOf { a: &a, b: &b };

        assert_eq!(c.rho(0.into()), (2,3));
        assert_eq!(c.rho(1.into()), (2,3));
        assert_eq!(c.rho(2.into()), (2,3));
        assert_eq!(c.rho(3.into()), (2,3));
        assert_eq!(c.rho(4.into()), END_EXTENT);
    }

    #[test]
    fn one_of_rho_nested_extents_with_trailing_extent() {
        let a = &[(1, 5)][..];
        let b = &[(2, 3), (6, 7)][..];
        let c = OneOf { a: &a, b: &b };

        assert_eq!(c.rho(4.into()), (6, 7));
    }

    #[test]
    fn one_of_rho_overlapping_extents() {
        let a = &[(1, 4), (2, 7)][..];
        let b = &[(3, 6)][..];
        let c = OneOf { a: &a, b: &b };

        assert_eq!(c.rho(4.into()), (1, 4));
    }

    #[test]
    fn one_of_rho_overlapping_and_nested_extents() {
        let a = &[(11, 78)][..];
        let b = &[(9, 60), (11, 136)][..];
        let c = OneOf { a: &a, b: &b };

        assert_eq!(c.rho(12.into()), (9, 60));
    }

    #[test]
    fn followed_by_all_tau_matches_all_rho() {
        fn prop(a: RandomExtentList, b: RandomExtentList) -> bool {
            let c = FollowedBy { a: &a, b: &b };
            iter_eq(c.iter_tau(), c.iter_rho())
        }

        quickcheck(prop as fn(RandomExtentList, RandomExtentList) -> bool);
    }

    #[test]
    fn followed_by_all_tau_prime_matches_all_rho_prime() {
        fn prop(a: RandomExtentList, b: RandomExtentList) -> bool {
            let c = FollowedBy { a: &a, b: &b };
            iter_eq(c.iter_tau_prime(), c.iter_rho_prime())
        }

        quickcheck(prop as fn(RandomExtentList, RandomExtentList) -> bool);
    }

    #[test]
    fn followed_by_any_k() {
        fn prop(a: RandomExtentList, b: RandomExtentList, k: Position) -> bool {
            any_k(FollowedBy { a: &a, b: &b }, k)
        }

        quickcheck(prop as fn(RandomExtentList, RandomExtentList, Position) -> bool);
    }

    #[test]
    fn followed_by_empty_lists() {
        let a = &[][..];
        let b = &[][..];
        let c = FollowedBy { a: a, b: b };
        assert_eq!(all_extents(c), []);
    }

    #[test]
    fn followed_by_one_empty_list() {
        let a = &[(1,2)][..];
        let b = &[][..];

        let c = FollowedBy { a: a, b: b };
        assert_eq!(all_extents(c), []);

        let c = FollowedBy { a: b, b: a };
        assert_eq!(all_extents(c), []);
    }

    #[test]
    fn followed_by_overlapping() {
        let a = &[(1,2)][..];
        let b = &[(2,3)][..];
        let c = FollowedBy { a: a, b: b };
        assert_eq!(all_extents(c), []);
    }

    #[test]
    fn followed_by_in_ascending_order() {
        let a = &[(1,2)][..];
        let b = &[(3,4)][..];
        let c = FollowedBy { a: a, b: b };
        assert_eq!(all_extents(c), [(1,4)]);
    }

    #[test]
    fn followed_by_in_descending_order() {
        let a = &[(3,4)][..];
        let b = &[(1,2)][..];
        let c = FollowedBy { a: a, b: b };
        assert_eq!(all_extents(c), []);
    }

    trait QuickcheckAlgebra : Algebra + Debug {
        fn clone_quickcheck_algebra(&self) -> Box<QuickcheckAlgebra + Send>;
    }

    impl<A> QuickcheckAlgebra for A
        where A: Algebra + Debug + Clone + Send + 'static
    {
        fn clone_quickcheck_algebra(&self) -> Box<QuickcheckAlgebra + Send> {
            Box::new(self.clone())
        }
    }

    #[derive(Debug)]
    struct ArbitraryAlgebraTree(Box<QuickcheckAlgebra + Send>);

    impl Clone for ArbitraryAlgebraTree {
        fn clone(&self) -> ArbitraryAlgebraTree {
            ArbitraryAlgebraTree(self.0.clone_quickcheck_algebra())
        }
    }

    impl Algebra for ArbitraryAlgebraTree {
        fn tau(&self, k: Position)       -> Extent { self.0.tau(k) }
        fn tau_prime(&self, k: Position) -> Extent { self.0.tau_prime(k) }
        fn rho(&self, k: Position)       -> Extent { self.0.rho(k) }
        fn rho_prime(&self, k: Position) -> Extent { self.0.rho_prime(k) }
    }

    impl Arbitrary for ArbitraryAlgebraTree {
        fn arbitrary<G>(g: &mut G) -> Self
            where G: quickcheck::Gen
        {
            let generate_node: bool = g.gen();

            if g.size() == 0 || ! generate_node {
                let extents = RandomExtentList::arbitrary(g);
                ArbitraryAlgebraTree(Box::new(extents))
            } else {
                let mut inner_gen = quickcheck::StdGen::new(rand::thread_rng(), g.size() / 2);

                let a = ArbitraryAlgebraTree::arbitrary(&mut inner_gen);
                let b = ArbitraryAlgebraTree::arbitrary(&mut inner_gen);

                let c: Box<QuickcheckAlgebra+Send> = match g.gen_range(0, 7) {
                    0 => Box::new(ContainedIn    { a: a, b: b }),
                    1 => Box::new(Containing     { a: a, b: b }),
                    2 => Box::new(NotContainedIn { a: a, b: b }),
                    3 => Box::new(NotContaining  { a: a, b: b }),
                    4 => Box::new(BothOf         { a: a, b: b }),
                    5 => Box::new(OneOf          { a: a, b: b }),
                    6 => Box::new(FollowedBy     { a: a, b: b }),
                    _ => unreachable!(),
                };

                ArbitraryAlgebraTree(c)
            }
        }
    }

    #[test]
    fn tree_of_operators_all_tau_matches_all_rho() {
        fn prop(a: ArbitraryAlgebraTree) -> bool {
            iter_eq((&a).iter_tau(), (&a).iter_rho())
        }

        quickcheck(prop as fn(ArbitraryAlgebraTree) -> bool);
    }

    #[test]
    fn tree_of_operators_all_tau_prime_matches_all_rho_prime() {
        fn prop(a: ArbitraryAlgebraTree) -> bool {
            iter_eq((&a).iter_tau_prime(), (&a).iter_rho_prime())
        }

        quickcheck(prop as fn(ArbitraryAlgebraTree) -> bool);
    }

    #[test]
    fn tree_of_operators_any_k() {
        fn prop(a: ArbitraryAlgebraTree, k: Position) -> bool {
            any_k(&a, k)
        }

        quickcheck(prop as fn(ArbitraryAlgebraTree, Position) -> bool);
    }
}

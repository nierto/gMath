//! One thread-local slot per constant type: the decimal engines' constants
//! (ln 2, pi, the exp tables) are built once per thread for each compute
//! word that is used, where 0.6.7 had one slot per profile.

use crate::fixed_point::domains::symbolic::rational::rational_number::OverflowDetected;
use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::rc::Rc;

thread_local! {
    static TYPED: RefCell<Vec<(TypeId, Rc<dyn Any>)>> = const { RefCell::new(Vec::new()) };
}

/// The thread's instance of `T`, built by `build` on first use.
pub(super) fn typed_cached<T: 'static>(build: impl FnOnce() -> Result<T, OverflowDetected>) -> Result<Rc<T>, OverflowDetected> {
    let id = TypeId::of::<T>();
    let found = TYPED.with(|v| v.borrow().iter().find(|(t, _)| *t == id).map(|(_, rc)| rc.clone()));
    if let Some(rc) = found {
        return Ok(rc.downcast::<T>().expect("typed cache: the key is the type"));
    }
    let rc = Rc::new(build()?);
    TYPED.with(|v| v.borrow_mut().push((id, rc.clone())));
    Ok(rc)
}

//! JavaScript-flavoured accessors over `serde_json::Value`.
//!
//! The Node projections are folds over loosely typed event payloads, and
//! the web client reads the folded state as JSON. Porting them onto typed
//! structs would either lose fields the client reads or force every optional
//! attribute into the type; keeping the records as JSON objects and porting
//! the *semantics* (`??`, `Number()`, truthiness, `undefined` versus `null`)
//! keeps the wire shape byte-for-byte and the handlers line-for-line. The
//! helpers here name those semantics once.

use serde_json::{Map, Number, Value};
use std::collections::HashMap;

pub type Obj = Map<String, Value>;

/// `value[key]` when present and not `null` (`??` semantics).
pub fn get<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    match value.get(key) {
        Some(Value::Null) | None => None,
        Some(v) => Some(v),
    }
}

/// `value[key]` when present, including `null`. `None` is `undefined`.
pub fn present<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    value.get(key)
}

pub fn get_str<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    get(value, key).and_then(Value::as_str)
}

pub fn get_bool(value: &Value, key: &str) -> Option<bool> {
    get(value, key).and_then(Value::as_bool)
}

pub fn get_arr<'a>(value: &'a Value, key: &str) -> Option<&'a Vec<Value>> {
    get(value, key).and_then(Value::as_array)
}

/// JavaScript truthiness.
pub fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0 && !f.is_nan()),
        Value::String(s) => !s.is_empty(),
        Value::Array(_) | Value::Object(_) => true,
    }
}

pub fn is_truthy(value: &Value, key: &str) -> bool {
    value.get(key).is_some_and(truthy)
}

/// `Number(value)`; `undefined` is passed as `None`.
pub fn number(value: Option<&Value>) -> f64 {
    match value {
        None => f64::NAN,
        Some(Value::Null) => 0.0,
        Some(Value::Bool(b)) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
        Some(Value::Number(n)) => n.as_f64().unwrap_or(f64::NAN),
        Some(Value::String(s)) => {
            let t = s.trim();
            if t.is_empty() {
                0.0
            } else {
                t.parse::<f64>().unwrap_or(f64::NAN)
            }
        }
        Some(Value::Array(_)) | Some(Value::Object(_)) => f64::NAN,
    }
}

/// `Number(obj[key])` for a record.
pub fn onum(obj: &Obj, key: &str) -> f64 {
    number(obj.get(key))
}

/// `Number(value[key])`.
pub fn num(value: &Value, key: &str) -> f64 {
    number(value.get(key))
}

/// `Number.isFinite(value[key]) ? value[key] : None` where the field is a
/// number (a string is not, matching `Number.isFinite`).
pub fn finite(value: &Value, key: &str) -> Option<f64> {
    match value.get(key) {
        Some(Value::Number(n)) => n.as_f64().filter(|f| f.is_finite()),
        _ => None,
    }
}

/// `Number.isInteger(x)`, for a number value.
pub fn is_integer(value: &Value) -> bool {
    match value {
        Value::Number(n) => n
            .as_f64()
            .is_some_and(|f| f.is_finite() && f.fract() == 0.0),
        _ => false,
    }
}

/// `String(value)`.
pub fn js_string(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => format_number(n.as_f64().unwrap_or(f64::NAN)),
        Value::String(s) => s.clone(),
        Value::Array(items) => items.iter().map(js_string).collect::<Vec<_>>().join(","),
        Value::Object(_) => "[object Object]".to_string(),
    }
}

/// A JavaScript number as text: integers without a fraction.
pub fn format_number(f: f64) -> String {
    if f.is_nan() {
        "NaN".to_string()
    } else if f.is_infinite() {
        if f > 0.0 { "Infinity" } else { "-Infinity" }.to_string()
    } else if f.fract() == 0.0 && f.abs() < 1e21 {
        format!("{}", f as i64)
    } else {
        format!("{f}")
    }
}

/// A JSON number from an f64, with integers stored as integers so the wire
/// text reads `3` rather than `3.0`. Non-finite values become `null`, as
/// `JSON.stringify` does.
pub fn jnum(f: f64) -> Value {
    if !f.is_finite() {
        return Value::Null;
    }
    if f.fract() == 0.0 && f.abs() < 9.0e15 {
        Value::Number(Number::from(f as i64))
    } else {
        Number::from_f64(f)
            .map(Value::Number)
            .unwrap_or(Value::Null)
    }
}

/// `Array.isArray(x) ? x.map(String) : []`.
pub fn strings(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .map(|items| items.iter().map(js_string).collect())
        .unwrap_or_default()
}

/// Set a key; `Some(v)` assigns (including `null`), `None` is `undefined`,
/// which `JSON.stringify` drops, so the key is removed.
pub fn assign(obj: &mut Obj, key: &str, value: Option<Value>) {
    match value {
        Some(v) => {
            obj.insert(key.to_string(), v);
        }
        None => {
            obj.remove(key);
        }
    }
}

pub fn set(obj: &mut Obj, key: &str, value: impl Into<Value>) {
    obj.insert(key.to_string(), value.into());
}

/// `a ?? b ?? ...`: the first present, non-null value.
pub fn coalesce<'a>(candidates: &[Option<&'a Value>]) -> Option<&'a Value> {
    candidates.iter().copied().flatten().find(|v| !v.is_null())
}

/// `x ?? null`.
pub fn or_null(value: Option<&Value>) -> Value {
    value.cloned().unwrap_or(Value::Null)
}

pub fn max_f(a: f64, b: f64) -> f64 {
    // JS Math.max: NaN poisons; callers guard with `finite` first.
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// `Math.round` (half away from zero for positives, as JS rounds half up).
pub fn round(f: f64) -> f64 {
    (f + 0.5).floor()
}

/// An insertion-ordered map, as a JavaScript `Map` is: iteration order is
/// insertion order, re-inserting an existing key keeps its position.
#[derive(Debug, Clone)]
pub struct OrderedMap<V> {
    order: Vec<String>,
    items: HashMap<String, V>,
}

impl<V> Default for OrderedMap<V> {
    fn default() -> Self {
        OrderedMap {
            order: Vec::new(),
            items: HashMap::new(),
        }
    }
}

impl<V> OrderedMap<V> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    pub fn contains(&self, key: &str) -> bool {
        self.items.contains_key(key)
    }

    pub fn get(&self, key: &str) -> Option<&V> {
        self.items.get(key)
    }

    pub fn get_mut(&mut self, key: &str) -> Option<&mut V> {
        self.items.get_mut(key)
    }

    pub fn insert(&mut self, key: impl Into<String>, value: V) {
        let key = key.into();
        if !self.items.contains_key(&key) {
            self.order.push(key.clone());
        }
        self.items.insert(key, value);
    }

    pub fn entry_or_insert_with(&mut self, key: &str, make: impl FnOnce() -> V) -> &mut V {
        if !self.items.contains_key(key) {
            self.order.push(key.to_string());
            self.items.insert(key.to_string(), make());
        }
        self.items.get_mut(key).expect("just inserted")
    }

    pub fn remove(&mut self, key: &str) -> Option<V> {
        let removed = self.items.remove(key);
        if removed.is_some() {
            self.order.retain(|k| k != key);
        }
        removed
    }

    pub fn clear(&mut self) {
        self.order.clear();
        self.items.clear();
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.order.iter()
    }

    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.order.iter().filter_map(move |k| self.items.get(k))
    }

    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut V> {
        let mut refs: Vec<*mut V> = Vec::with_capacity(self.order.len());
        for key in &self.order {
            if let Some(v) = self.items.get_mut(key) {
                refs.push(v as *mut V);
            }
        }
        // SAFETY: every pointer names a distinct value owned by `self.items`,
        // which is borrowed mutably for the iterator's whole lifetime.
        refs.into_iter().map(|p| unsafe { &mut *p })
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &V)> {
        self.order
            .iter()
            .filter_map(move |k| self.items.get(k).map(|v| (k, v)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn number_coercion_follows_javascript() {
        assert_eq!(number(Some(&json!("4"))), 4.0);
        assert_eq!(number(Some(&json!(""))), 0.0);
        assert!(number(Some(&json!("nope"))).is_nan());
        assert!(number(None).is_nan());
        assert_eq!(number(Some(&Value::Null)), 0.0);
        assert_eq!(number(Some(&json!(true))), 1.0);
        assert!(number(Some(&json!({}))).is_nan());
    }

    #[test]
    fn strings_and_numbers_print_like_javascript() {
        assert_eq!(js_string(&json!(3.0)), "3");
        assert_eq!(js_string(&json!(2.5)), "2.5");
        assert_eq!(js_string(&Value::Null), "null");
        assert_eq!(jnum(3.0), json!(3));
        assert_eq!(jnum(f64::NAN), Value::Null);
    }

    #[test]
    fn the_ordered_map_keeps_insertion_order_across_reinsert_and_remove() {
        let mut m = OrderedMap::new();
        m.insert("b", 1);
        m.insert("a", 2);
        m.insert("b", 3);
        m.insert("c", 4);
        m.remove("a");
        let keys: Vec<&String> = m.keys().collect();
        assert_eq!(keys, ["b", "c"]);
        assert_eq!(m.values().copied().collect::<Vec<_>>(), [3, 4]);
        for v in m.values_mut() {
            *v += 10;
        }
        assert_eq!(m.get("b"), Some(&13));
    }
}

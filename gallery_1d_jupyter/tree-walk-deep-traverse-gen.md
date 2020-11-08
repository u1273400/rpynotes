# JavaScript Generator Function to Deep Traverse JSON-Encodable Data

## Helpers

```js
const isArray = (value) => Array.isArray(value);

const isObject = (value) =>
  Object.prototype.toString.call(value) === '[object Object]';
```

## `walkTree`

```js
function* walkTree(collection) {
  function* visit(value, path, parentNode) {
    const node = Object.create({ parentNode });
    node.path = path;
    node.value = value;
    yield node;
    if (isArray(value)) {
      for (let i = 0, len = value.length; i < len; i++) {
        yield* visit(value[i], path.concat(i), node);
      }
    } else if (isObject(value)) {
      for (const key in value) {
        if (value.hasOwnProperty(key)) {
          yield* visit(value[key], path.concat(key), node);
        }
      }
    }
  }
  yield* visit(collection, [], null);
}
```

### Example

```js
const treeWalker = walkTree({ a: { b: { c: [1, 2] } } });

for (const value of treeWalker) {
  console.log(value, value.parentNode);
}
```

Output:

```
{ path: [], value: { a: { b: [Object] } } } null
{ path: [ 'a' ], value: { b: { c: [Array] } } } { path: [], value: { a: { b: [Object] } } }
{ path: [ 'a', 'b' ], value: { c: [ 1, 2 ] } } { path: [ 'a' ], value: { b: { c: [Array] } } }
{ path: [ 'a', 'b', 'c' ], value: [ 1, 2 ] } { path: [ 'a', 'b' ], value: { c: [ 1, 2 ] } }
{ path: [ 'a', 'b', 'c', 0 ], value: 1 } { path: [ 'a', 'b', 'c' ], value: [ 1, 2 ] }
{ path: [ 'a', 'b', 'c', 1 ], value: 2 } { path: [ 'a', 'b', 'c' ], value: [ 1, 2 ] }
```


## walkTreeWithTypes

```js
function* walkTreeWithTypes(collection) {
  const treeWalker = walkTree(collection);
  let next;
  while ((next = treeWalker.next()) && !next.done) {
    const node = next.value;
    if (isArray(node.value)) {
      node.isArray = true;
      node.isBranch = true;
    } else if (isObject(node.value)) {
      node.isBranch = true;
      node.isObject = true;
    } else {
      node.isLeaf = true;
    }
    yield node;
  }
}
```

### Example

```js
const typesTreeWalker = walkTreeWithTypes({ a: { b: { c: [1, 2] } } });
for (const value of typesTreeWalker) {
  console.log(value, value.parentNode);
}
```

Output:

```
{ path: [],
  value: { a: { b: [Object] } },
  isBranch: true,
  isObject: true } null
{ path: [ 'a' ],
  value: { b: { c: [Array] } },
  isBranch: true,
  isObject: true } { path: [],
  value: { a: { b: [Object] } },
  isBranch: true,
  isObject: true }
{ path: [ 'a', 'b' ],
  value: { c: [ 1, 2 ] },
  isBranch: true,
  isObject: true } { path: [ 'a' ],
  value: { b: { c: [Array] } },
  isBranch: true,
  isObject: true }
{ path: [ 'a', 'b', 'c' ],
  value: [ 1, 2 ],
  isArray: true,
  isBranch: true } { path: [ 'a', 'b' ],
  value: { c: [ 1, 2 ] },
  isBranch: true,
  isObject: true }
{ path: [ 'a', 'b', 'c', 0 ], value: 1, isLeaf: true } { path: [ 'a', 'b', 'c' ],
  value: [ 1, 2 ],
  isArray: true,
  isBranch: true }
{ path: [ 'a', 'b', 'c', 1 ], value: 2, isLeaf: true } { path: [ 'a', 'b', 'c' ],
  value: [ 1, 2 ],
  isArray: true,
  isBranch: true }
```

# An improvement to Bridson's Algorithm for Poisson Disc sampling.

Improved Version of Bridson's Algorithm for Poisson Disc Sampling [here](https://observablehq.com/@techsparx/an-improvement-on-bridsons-algorithm-for-poisson-disc-samp/2) by Martin Roberts.
Original Algorithm [here](https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf).

In many applications in graphics, particularly rendering, generating samples from a blue noise distribution is 
important. Poisson-disc sampling produces points that are tightly-packed, but no closer to each other than a specified 
minimum distance, resulting in a more natural pattern.

To create such distributions, most people use Mitchell's best candidate, however, Bridson's 2007 algorithm is much 
faster as it is only O(n) time, rather than O(n^2) for Mitchell's.

However, I show how, by changing only a few lines of code, we can make Bridson's algorithm much more efficient (ie, 
about 20x faster!).

This new version is not only much faster, but it produces higher quality point distributions, as it allows for more 
tightly packed and consistent point distributions.

The animation below illustrates how the frontier of candidate points is selected in this new algorithm. Comparing 
this with Mike Bostock's visualization of the Bridson algorithm, makes it clear how these two algorithms differ.

See also Jason Davies' implementation of Bridson's original algorithm which uses Bridson’s original 2007 algorithm.

```javascript
poissonDiscSampler = ƒ*(width, height, radius)
```

```javascript
function* poissonDiscSampler(width, height, radius) {
  const k = 4; // maximum number of samples before rejection
  const radius2 = radius * radius;
  const cellSize = radius * Math.SQRT1_2;
  const gridWidth = Math.ceil(width / cellSize);
  const gridHeight = Math.ceil(height / cellSize);
  const grid = new Array(gridWidth * gridHeight);
  const queue = [];

  // Pick the first sample.
  yield {add: sample(width / 2 , height / 2, null)};

  // Pick a random existing sample from the queue.
  pick: while (queue.length) {
    const i = Math.random() * queue.length | 0;
    const parent = queue[i];
    const seed = Math.random();
    const epsilon = 0.0000001;
    
    // Make a new candidate.
    for (let j = 0; j < k; ++j) {
      const a = 2 * Math.PI * (seed + 1.0*j/k);
      const r = radius + epsilon;
      const x = parent[0] + r * Math.cos(a);
      const y = parent[1] + r * Math.sin(a);

      // Accept candidates that are inside the allowed extent
      // and farther than 2 * radius to all existing samples.
      if (0 <= x && x < width && 0 <= y && y < height && far(x, y)) {
        yield {add: sample(x, y), parent};
        continue pick;
      }
    }

    // If none of k candidates were accepted, remove it from the queue.
    const r = queue.pop();
    if (i < queue.length) queue[i] = r;
    yield {remove: parent};
  }

  function far(x, y) {
    const i = x / cellSize | 0;
    const j = y / cellSize | 0;
    const i0 = Math.max(i - 2, 0);
    const j0 = Math.max(j - 2, 0);
    const i1 = Math.min(i + 3, gridWidth);
    const j1 = Math.min(j + 3, gridHeight);
    for (let j = j0; j < j1; ++j) {
      const o = j * gridWidth;
      for (let i = i0; i < i1; ++i) {
        const s = grid[o + i];
        if (s) {
          const dx = s[0] - x;
          const dy = s[1] - y;
          if (dx * dx + dy * dy < radius2) return false;
        }
      }
    }
    return true;
  }

  function sample(x, y, parent) {
    const s = grid[gridWidth * (y / cellSize | 0) + (x / cellSize | 0)] = [x, y];
    queue.push(s);
    return s;
  }
}
```
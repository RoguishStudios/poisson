//
// Copyright 2020-2025 Hans W. Uhlig. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

//! # Poisson Disc Sampling
//!
//! This crate provides an efficient implementation of Bridson's algorithm for generating
//! Poisson disc distributions. Poisson disc sampling creates tightly-packed point distributions
//! where no two points are closer than a specified minimum radius, resulting in natural-looking
//! blue noise patterns.
//!
//! ## Algorithm
//!
//! This implementation uses an improved version of Bridson's 2007 algorithm, which runs in
//! O(n) time. The algorithm maintains a spatial grid for efficient neighbor queries and
//! an active queue of candidate points for expansion.
//!
//! ## Usage
//!
//! ```rust
//! use rand::prelude::*;
//! use poisson::{PoissonDiscSampler, SamplerEvent};
//!
//! let mut sampler = PoissonDiscSampler::builder(rand::rng())
//!     .width(500)
//!     .height(500)
//!     .radius(10.0)
//!     .build();
//!
//! loop {
//!     match sampler.step() {
//!         SamplerEvent::Created(point) => {
//!             println!("New point at ({}, {})", point.x, point.y);
//!         }
//!         SamplerEvent::Closed(point) => {
//!             // Point exhausted - no more valid neighbors
//!         }
//!         SamplerEvent::Complete => break,
//!     }
//! }
//! ```
//!
//! ## Features
//!
//! - **Efficient**: O(n) time complexity with spatial grid acceleration
//! - **Step-by-step generation**: Allows incremental point generation with event tracking
//! - **Configurable**: Control density, radius, and sampling behavior
//! - **Deterministic**: Same RNG seed produces identical results

#![warn(
    clippy::cargo,
    missing_docs,
    clippy::pedantic,
    future_incompatible,
    rust_2018_idioms
)]

use rand::prelude::Rng;

/// Events emitted during the Poisson disc sampling process.
///
/// Each call to [`PoissonDiscSampler::step`] returns an event indicating
/// the current state of the sampling process.
#[derive(Clone, Debug)]
pub enum SamplerEvent {
    /// A new point was successfully created and added to the distribution.
    ///
    /// The contained [`SamplerPoint`] represents the coordinates of the newly created point.
    Created(SamplerPoint),

    /// A point was closed because it could not generate any more valid neighbors.
    ///
    /// The contained [`SamplerPoint`] represents the coordinates of the closed point.
    /// This happens when all attempts to place new points around this location fail
    /// due to being too close to existing points or falling outside the bounds.
    Closed(SamplerPoint),

    /// The sampling process has completed.
    ///
    /// No more points can be generated. The active queue is empty and all possible
    /// locations have been exhausted.
    Complete,
}

/// A 2D point in the Poisson disc distribution.
///
/// Represents a location in continuous 2D space with floating-point coordinates.
#[derive(Clone, Debug, Default, PartialOrd, PartialEq)]
pub struct SamplerPoint {
    /// The x-coordinate of the point.
    pub x: f64,
    /// The y-coordinate of the point.
    pub y: f64,
}

/// A Poisson disc sampler that generates well-distributed points.
///
/// This sampler implements an improved version of Bridson's algorithm for generating
/// Poisson disc distributions. Points are generated incrementally via the [`step`](Self::step)
/// method, allowing observation of the generation process.
///
/// # Type Parameters
///
/// * `R` - A random number generator implementing the [`Rng`] trait
///
/// # Grid-based Acceleration
///
/// The sampler divides the sample space into a grid of cells for efficient distance
/// queries. Each cell has a size of `radius * sqrt(0.5)`, ensuring that checking a
/// 5×5 neighborhood around a point is sufficient to verify the minimum distance constraint.
#[derive(Clone)]
pub struct PoissonDiscSampler<R: Rng> {
    /// Whether to use dense packing (points exactly `radius` apart) or allow some variation.
    dense: bool,
    /// The current iteration/round number.
    round: u64,
    /// Width of the sampling area.
    width: u64,
    /// Height of the sampling area.
    height: u64,
    /// Minimum distance between any two points.
    radius: f64,
    /// Squared radius for faster distance comparisons (avoids sqrt).
    radius2: f64,
    /// Size of each grid cell (radius * sqrt(0.5)).
    cell_size: f64,
    /// Number of grid cells in the horizontal direction.
    grid_width: i64,
    /// Number of grid cells in the vertical direction.
    grid_height: i64,
    /// Maximum number of candidate attempts per active point before removal.
    max_samples: usize,
    /// Spatial grid storing points for efficient neighbor queries.
    grid: Vec<Option<SamplerPoint>>,
    /// Active queue of points that may still generate valid neighbors.
    queue: Vec<SamplerPoint>,
    /// Random number generator for point generation.
    rng: R,
}

impl<R: Rng> PoissonDiscSampler<R> {
    /// Creates a new builder for configuring a Poisson disc sampler.
    ///
    /// # Arguments
    ///
    /// * `rng` - A random number generator implementing [`Rng`]
    ///
    /// # Returns
    ///
    /// A [`Builder`] instance with default values that can be customized before building.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use poisson::PoissonDiscSampler;
    ///
    /// let sampler = PoissonDiscSampler::builder(rand::rng())
    ///     .width(800)
    ///     .height(600)
    ///     .radius(15.0)
    ///     .build();
    /// ```
    pub fn builder(rng: R) -> Builder<R> {
        Builder {
            dense: true,
            width: 100,
            height: 100,
            radius: 10.0,
            max_samples: 4,
            rng,
        }
    }

    /// Advances the sampler by one step and returns the resulting event.
    ///
    /// This method should be called repeatedly until it returns [`SamplerEvent::Complete`].
    /// Each call either creates a new point, closes an exhausted point, or signals completion.
    ///
    /// # Algorithm
    ///
    /// 1. On the first call (round 0), places the initial point at the center
    /// 2. While the queue is non-empty:
    ///    - Randomly selects an active point from the queue
    ///    - Attempts to place `max_samples` candidate points around it in an annulus
    ///    - If a valid candidate is found, adds it and returns [`SamplerEvent::Created`]
    ///    - If all attempts fail, removes the point from the queue and returns [`SamplerEvent::Closed`]
    /// 3. When the queue is empty, returns [`SamplerEvent::Complete`]
    ///
    /// # Returns
    ///
    /// A [`SamplerEvent`] indicating what happened in this step.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// loop {
    ///     match sampler.step() {
    ///         SamplerEvent::Created(point) => println!("Created: {:?}", point),
    ///         SamplerEvent::Closed(point) => println!("Closed: {:?}", point),
    ///         SamplerEvent::Complete => break,
    ///     }
    /// }
    /// ```
    pub fn step(&mut self) -> SamplerEvent {
        if self.round == 0 {
            self.round += 1;
            return SamplerEvent::Created(
                self.sample(self.width as f64 / 2.0, self.height as f64 / 2.0),
            );
        } else if !self.queue.is_empty() {
            self.round += 1;
            let i = self.rng.random_range(0..self.queue.len());
            let parent = self.queue[i].clone();
            let seed = self.rng.random::<f64>();

            // Make a new candidate.
            for j in 0..self.max_samples {
                let a = 2.0 * std::f64::consts::PI * (seed + (j as f64 / self.max_samples as f64));
                let r = if self.dense {
                    self.radius + f64::EPSILON
                } else {
                    self.radius + self.rng.random_range(0.0..self.radius)
                };
                let x = parent.x + r * f64::cos(a);
                let y = parent.y + r * f64::sin(a);

                // Accept candidates that are inside the allowed extent
                // and farther than 2 * radius to all existing samples.
                if (0.0 <= x && x < self.width as f64)
                    && (0.0 <= y && y < self.height as f64)
                    && self.far(x, y)
                {
                    return SamplerEvent::Created(self.sample(x, y));
                }
            }

            // If none of k candidates were accepted, remove it from the queue.
            SamplerEvent::Closed(self.queue.swap_remove(i))
        } else {
            SamplerEvent::Complete
        }
    }

    /// Checks if a candidate point is far enough from all existing points.
    ///
    /// Uses the spatial grid to efficiently check only nearby cells rather than
    /// all existing points. A 5×5 neighborhood of cells is sufficient due to the
    /// cell size being `radius * sqrt(0.5)`.
    ///
    /// # Arguments
    ///
    /// * `x` - The x-coordinate of the candidate point
    /// * `y` - The y-coordinate of the candidate point
    ///
    /// # Returns
    ///
    /// `true` if the point is at least `radius` distance from all existing points,
    /// `false` otherwise.
    fn far(&mut self, x: f64, y: f64) -> bool {
        let i = f64::floor(x / self.cell_size) as i64;
        let j = f64::floor(y / self.cell_size) as i64;
        let i0 = i64::max(i - 2, 0);
        let j0 = i64::max(j - 2, 0);
        let i1 = i64::min(i + 3, self.grid_width as i64);
        let j1 = i64::min(j + 3, self.grid_height as i64);

        for j in j0..j1 {
            let o = j * self.grid_width;
            for i in i0..i1 {
                let idx = (o + i) as usize;
                if let Some(s) = &self.grid[idx] {
                    let dx = s.x - x;
                    let dy = s.y - y;
                    if dx * dx + dy * dy < self.radius2 {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Adds a point to the distribution at the specified coordinates.
    ///
    /// This internal method updates the spatial grid and active queue with the new point.
    ///
    /// # Arguments
    ///
    /// * `x` - The x-coordinate of the new point
    /// * `y` - The y-coordinate of the new point
    ///
    /// # Returns
    ///
    /// The newly created [`SamplerPoint`].
    fn sample(&mut self, x: f64, y: f64) -> SamplerPoint {
        let point = SamplerPoint { x, y };
        let grid_x = f64::floor(x / self.cell_size) as i64;
        let grid_y = f64::floor(y / self.cell_size) as i64;
        let grid_idx = grid_y * self.grid_width + grid_x;
        self.grid[grid_idx as usize] = Some(point.clone());
        self.queue.push(point.clone());
        point
    }
}

impl<R: Rng> std::fmt::Debug for PoissonDiscSampler<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PoissonDiscSampler")
            .field("dense", &self.dense)
            .field("round", &self.round)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("radius", &self.radius)
            .field("cell_size", &self.cell_size)
            .field("grid_width", &self.grid_width)
            .field("grid_height", &self.grid_height)
            .field("max_samples", &self.max_samples)
            .field("queue_size", &self.queue.len())
            .finish()
    }
}

/// Builder for configuring and creating a [`PoissonDiscSampler`].
///
/// Provides a fluent interface for setting sampler parameters before construction.
/// Default values provide a reasonable starting configuration.
///
/// # Type Parameters
///
/// * `R` - A random number generator implementing the [`Rng`] trait
pub struct Builder<R: Rng> {
    /// Whether to use dense packing mode.
    dense: bool,
    /// Width of the sampling area.
    width: u64,
    /// Height of the sampling area.
    height: u64,
    /// Minimum distance between points.
    radius: f64,
    /// Maximum number of candidate attempts per point.
    max_samples: usize,
    /// Random number generator.
    rng: R,
}

impl<R: Rng> Builder<R> {
    /// Sets the width of the sampling area.
    ///
    /// # Arguments
    ///
    /// * `width` - The width in coordinate units (default: 100)
    ///
    /// # Returns
    ///
    /// The builder for method chaining.
    pub fn width(mut self, width: u64) -> Self {
        self.width = width;
        self
    }

    /// Sets the height of the sampling area.
    ///
    /// # Arguments
    ///
    /// * `height` - The height in coordinate units (default: 100)
    ///
    /// # Returns
    ///
    /// The builder for method chaining.
    pub fn height(mut self, height: u64) -> Self {
        self.height = height;
        self
    }

    /// Sets the minimum distance between any two points.
    ///
    /// # Arguments
    ///
    /// * `radius` - The minimum separation distance (default: 10.0)
    ///
    /// # Returns
    ///
    /// The builder for method chaining.
    pub fn radius(mut self, radius: f64) -> Self {
        self.radius = radius;
        self
    }

    /// Sets whether to use dense packing mode.
    ///
    /// In dense mode, new points are placed exactly `radius` distance from their parent.
    /// In non-dense mode, points are placed between `radius` and `2 * radius` away,
    /// allowing more variation in the distribution.
    ///
    /// # Arguments
    ///
    /// * `dense` - `true` for dense packing, `false` for variable spacing (default: true)
    ///
    /// # Returns
    ///
    /// The builder for method chaining.
    pub fn dense(mut self, dense: bool) -> Self {
        self.dense = dense;
        self
    }

    /// Sets the maximum number of candidate attempts per active point.
    ///
    /// Higher values may produce slightly better distributions at the cost of more
    /// computation. Bridson's paper suggests k=30, but k=4 often provides good results.
    ///
    /// # Arguments
    ///
    /// * `max_samples` - Number of candidate attempts (default: 4)
    ///
    /// # Returns
    ///
    /// The builder for method chaining.
    pub fn max_samples(mut self, max_samples: usize) -> Self {
        self.max_samples = max_samples;
        self
    }

    /// Constructs the configured [`PoissonDiscSampler`].
    ///
    /// Initializes the spatial grid and prepares the sampler for point generation.
    ///
    /// # Returns
    ///
    /// A fully configured [`PoissonDiscSampler`] ready to generate points.
    pub fn build(self) -> PoissonDiscSampler<R> {
        let cell_size = self.radius * f64::sqrt(0.5);
        let grid_width = f64::ceil(self.width as f64 / cell_size) as i64;
        let grid_height = f64::ceil(self.height as f64 / cell_size) as i64;
        PoissonDiscSampler {
            max_samples: self.max_samples,
            round: 0,
            width: self.width,
            height: self.height,
            radius: self.radius,
            radius2: self.radius * self.radius,
            dense: self.dense,
            cell_size,
            grid_width,
            grid_height,
            grid: vec![None; (grid_width * grid_height) as usize],
            queue: Vec::default(),
            rng: self.rng,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_xoshiro::rand_core::SeedableRng;
    use rand_xoshiro::Xoroshiro128PlusPlus;

    /// Helper function to collect all generated points from a sampler
    fn collect_all_points<R: Rng>(sampler: &mut PoissonDiscSampler<R>) -> Vec<SamplerPoint> {
        let mut points = Vec::new();
        loop {
            match sampler.step() {
                SamplerEvent::Created(point) => points.push(point),
                SamplerEvent::Closed(_) => {}
                SamplerEvent::Complete => break,
            }
        }
        points
    }

    /// Helper function to calculate distance between two points
    fn distance(p1: &SamplerPoint, p2: &SamplerPoint) -> f64 {
        let dx = p1.x - p2.x;
        let dy = p1.y - p2.y;
        (dx * dx + dy * dy).sqrt()
    }

    #[test]
    fn test_basic_generation() {
        let rng = Xoroshiro128PlusPlus::from_seed([0; 16]);
        let mut sampler = PoissonDiscSampler::builder(rng)
            .width(1000)
            .height(1000)
            .radius(10.0)
            .build();
        let points = collect_all_points(&mut sampler);

        assert!(!points.is_empty(), "Should generate at least one point");
    }

    #[test]
    fn test_first_point_centered() {
        let rng = Xoroshiro128PlusPlus::from_seed([0; 16]);
        let mut sampler = PoissonDiscSampler::builder(rng)
            .width(100)
            .height(100)
            .radius(10.0)
            .build();

        if let SamplerEvent::Created(point) = sampler.step() {
            assert_eq!(point.x, 50.0, "First point should be centered horizontally");
            assert_eq!(point.y, 50.0, "First point should be centered vertically");
        } else {
            panic!("First step should create a point");
        }
    }

    #[test]
    fn test_minimum_radius_constraint() {
        let rng = Xoroshiro128PlusPlus::from_seed([42; 16]);
        let radius = 15.0;
        let mut sampler = PoissonDiscSampler::builder(rng)
            .width(500)
            .height(500)
            .radius(radius)
            .build();
        let points = collect_all_points(&mut sampler);

        // Check that all points are at least `radius` apart
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                let dist = distance(&points[i], &points[j]);
                assert!(
                    dist >= radius - f64::EPSILON,
                    "Points {} and {} are too close: distance = {}, radius = {}",
                    i,
                    j,
                    dist,
                    radius
                );
            }
        }
    }

    #[test]
    fn test_points_within_bounds() {
        let rng = Xoroshiro128PlusPlus::from_seed([123; 16]);
        let width = 200;
        let height = 150;
        let mut sampler = PoissonDiscSampler::builder(rng)
            .width(width)
            .height(height)
            .radius(10.0)
            .build();
        let points = collect_all_points(&mut sampler);

        for (idx, point) in points.iter().enumerate() {
            assert!(
                point.x >= 0.0 && point.x < width as f64,
                "Point {} x-coordinate {} is out of bounds [0, {})",
                idx,
                point.x,
                width
            );
            assert!(
                point.y >= 0.0 && point.y < height as f64,
                "Point {} y-coordinate {} is out of bounds [0, {})",
                idx,
                point.y,
                height
            );
        }
    }

    #[test]
    fn test_small_area_generation() {
        let rng = Xoroshiro128PlusPlus::from_seed([1; 16]);
        let mut sampler = PoissonDiscSampler::builder(rng)
            .width(50)
            .height(50)
            .radius(5.0)
            .build();
        let points = collect_all_points(&mut sampler);

        assert!(
            !points.is_empty(),
            "Should generate points even in small area"
        );
        assert!(
            points.len() < 100,
            "Should not generate too many points in small area"
        );
    }

    #[test]
    fn test_large_radius_limits_points() {
        let rng = Xoroshiro128PlusPlus::from_seed([2; 16]);
        let mut sampler = PoissonDiscSampler::builder(rng)
            .width(100)
            .height(100)
            .radius(40.0)
            .build();
        let points = collect_all_points(&mut sampler);

        // With a large radius relative to the area, we should get very few points
        assert!(
            points.len() < 10,
            "Large radius should limit point count, got {}",
            points.len()
        );
    }

    #[test]
    fn test_small_radius_generates_many_points() {
        let rng = Xoroshiro128PlusPlus::from_seed([3; 16]);
        let mut sampler = PoissonDiscSampler::builder(rng)
            .width(500)
            .height(500)
            .radius(5.0)
            .dense(false)
            .build();
        let points = collect_all_points(&mut sampler);

        // Small radius should allow many points
        assert!(
            points.len() > 100,
            "Small radius should generate many points, got {}",
            points.len()
        );
    }

    #[test]
    fn test_event_sequence() {
        let rng = Xoroshiro128PlusPlus::from_seed([4; 16]);
        let mut sampler = PoissonDiscSampler::builder(rng)
            .width(100)
            .height(100)
            .radius(10.0)
            .build();

        let mut created_count = 0;
        let mut closed_count = 0;
        let mut complete_count = 0;

        loop {
            match sampler.step() {
                SamplerEvent::Created(_) => created_count += 1,
                SamplerEvent::Closed(_) => closed_count += 1,
                SamplerEvent::Complete => {
                    complete_count += 1;
                    break;
                }
            }
        }

        assert!(created_count > 0, "Should create at least one point");
        assert_eq!(complete_count, 1, "Should complete exactly once");
        assert!(
            closed_count <= created_count,
            "Some points may remain open at completion"
        );
    }

    #[test]
    fn test_deterministic_with_same_seed() {
        let rng1 = Xoroshiro128PlusPlus::from_seed([100; 16]);
        let mut sampler1 = PoissonDiscSampler::builder(rng1)
            .width(200)
            .height(200)
            .radius(10.0)
            .build();
        let points1 = collect_all_points(&mut sampler1);

        let rng2 = Xoroshiro128PlusPlus::from_seed([100; 16]);
        let mut sampler2 = PoissonDiscSampler::builder(rng2)
            .width(200)
            .height(200)
            .radius(10.0)
            .build();
        let points2 = collect_all_points(&mut sampler2);

        assert_eq!(
            points1.len(),
            points2.len(),
            "Same seed should produce same number of points"
        );

        for (i, (p1, p2)) in points1.iter().zip(points2.iter()).enumerate() {
            assert!(
                (p1.x - p2.x).abs() < f64::EPSILON && (p1.y - p2.y).abs() < f64::EPSILON,
                "Point {} differs: ({}, {}) vs ({}, {})",
                i,
                p1.x,
                p1.y,
                p2.x,
                p2.y
            );
        }
    }

    #[test]
    fn test_different_seeds_produce_different_results() {
        let rng1 = Xoroshiro128PlusPlus::from_seed([1; 16]);
        let mut sampler1 = PoissonDiscSampler::builder(rng1)
            .width(200)
            .height(200)
            .radius(10.0)
            .build();
        let points1 = collect_all_points(&mut sampler1);

        let rng2 = Xoroshiro128PlusPlus::from_seed([2; 16]);
        let mut sampler2 = PoissonDiscSampler::builder(rng2)
            .width(200)
            .height(200)
            .radius(10.0)
            .build();
        let points2 = collect_all_points(&mut sampler2);

        // First point is always at center, so skip it
        // Different seeds should produce different distributions
        let different_count = points1
            .len()
            .ne(&points2.len())
            .then_some(1)
            .unwrap_or(0);

        let mut point_differences = 0;
        let min_len = points1.len().min(points2.len());

        // Skip the first point since it's always deterministic (center)
        for i in 1..min_len {
            if (points1[i].x - points2[i].x).abs() > f64::EPSILON
                || (points1[i].y - points2[i].y).abs() > f64::EPSILON {
                point_differences += 1;
            }
        }

        assert!(
            different_count + point_differences > 0,
            "Different seeds should produce different point distributions (different count or different points)"
        );
    }

    #[test]
    fn test_rectangular_area() {
        let rng = Xoroshiro128PlusPlus::from_seed([5; 16]);
        let mut sampler = PoissonDiscSampler::builder(rng)
            .width(1000)
            .height(100)
            .radius(10.0)
            .build();
        let points = collect_all_points(&mut sampler);

        assert!(
            !points.is_empty(),
            "Should generate points in rectangular area"
        );

        // Verify all points respect the dimensions
        for point in &points {
            assert!(point.x >= 0.0 && point.x < 1000.0);
            assert!(point.y >= 0.0 && point.y < 100.0);
        }
    }

    #[test]
    fn test_sampler_point_default() {
        let point = SamplerPoint::default();
        assert_eq!(point.x, 0.0);
        assert_eq!(point.y, 0.0);
    }

    #[test]
    fn test_sampler_point_clone() {
        let point1 = SamplerPoint { x: 10.5, y: 20.3 };
        let point2 = point1.clone();
        assert_eq!(point1.x, point2.x);
        assert_eq!(point1.y, point2.y);
    }

    #[test]
    fn test_debug_output() {
        let rng = Xoroshiro128PlusPlus::from_seed([0; 16]);
        let sampler = PoissonDiscSampler::builder(rng)
            .width(100)
            .height(100)
            .radius(10.0)
            .build();
        let debug_str = format!("{:?}", sampler);

        assert!(debug_str.contains("PoissonDiscSampler"));
        assert!(debug_str.contains("width"));
        assert!(debug_str.contains("height"));
        assert!(debug_str.contains("radius"));
    }

    #[test]
    fn test_very_small_area() {
        let rng = Xoroshiro128PlusPlus::from_seed([6; 16]);
        let mut sampler = PoissonDiscSampler::builder(rng)
            .width(20)
            .height(20)
            .radius(5.0)
            .build();
        let points = collect_all_points(&mut sampler);

        // Should still generate at least the center point
        assert!(
            !points.is_empty(),
            "Should generate at least one point in tiny area"
        );
    }

    #[test]
    fn test_radius_approximately_maintained() {
        let rng = Xoroshiro128PlusPlus::from_seed([7; 16]);
        let radius = 20.0;
        let mut sampler = PoissonDiscSampler::builder(rng)
            .width(400)
            .height(400)
            .radius(radius)
            .build();
        let points = collect_all_points(&mut sampler);

        // Calculate minimum distance found
        let mut min_distance = f64::INFINITY;
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                let dist = distance(&points[i], &points[j]);
                min_distance = min_distance.min(dist);
            }
        }

        assert!(
            min_distance >= radius - f64::EPSILON,
            "Minimum distance {} should be at least radius {}",
            min_distance,
            radius
        );
    }

    #[test]
    fn test_coverage_density() {
        let rng = Xoroshiro128PlusPlus::from_seed([8; 16]);
        let width = 500;
        let height = 500;
        let radius = 10.0;
        let mut sampler = PoissonDiscSampler::builder(rng)
            .width(width)
            .height(height)
            .radius(radius)
            .dense(false)
            .build();
        let points = collect_all_points(&mut sampler);

        // For Poisson disc sampling with minimum separation r, the theoretical maximum
        // is approximately area / (r²), which represents a square grid packing.
        // In practice, Poisson disc sampling achieves less than this due to randomness.
        let area = (width * height) as f64;
        let max_theoretical = area / (radius * radius);

        assert!(
            (points.len() as f64) < max_theoretical,
            "Point count of {} should be less than the theoretical maximum of {}",
            points.len(),
            max_theoretical
        );

        // Poisson disc sampling typically achieves 50-80% of square grid density
        let min_expected = max_theoretical * 0.5;
        assert!(
            (points.len() as f64) > min_expected,
            "Point count of {} should be at least {} (50% of theoretical)",
            points.len(),
            min_expected
        );
    }
}

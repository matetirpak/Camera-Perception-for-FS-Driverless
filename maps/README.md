# maps

Cone maps as CSV, which FSDS loads with `-CustomMapPath` (`scripts/sim.sh start maps/<file>.csv`). Columns are
`tag,x,y,direction,x_variance,y_variance,xy_covariance`; the tag is `blue`, `yellow` or `big_orange`, x and y
are metres in ENU, and the rest is ignored by the loader. The car spawns at the origin facing +x and the
ground truth track equals the file.

- `depth_test.csv`: eight small cones straight ahead at 3 to 20 m, alternating left and right, for the
  standing depth test of [docs/mono.md](../docs/mono.md) (Gosala et al., Fig. 9a). The two big orange cones
  the loader needs for its start line sit behind the car.

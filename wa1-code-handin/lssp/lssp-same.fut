-- Parallel Longest Satisfying Segment
--
-- ==
-- compiled input {
--    [1i32, -2i32, -2i32, 0i32, 0i32, 0i32, 0i32, 0i32, 3i32, 4i32, -6i32, 1i32]
-- }
-- output {
--    5i32
-- }
--
-- compiled input {
--   [3i32, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2]
-- }
-- output {
--   8
-- }
--
-- compiled input {
--   [1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10]
-- }
-- output {
--  1
-- }


import "lssp"
import "lssp-seq"

let main (xs: []i32) : i32 =
  let pred1 _x = true
  let pred2 x y = (x == y)
--  in  lssp_seq pred1 pred2 xs
  in  lssp pred1 pred2 xs

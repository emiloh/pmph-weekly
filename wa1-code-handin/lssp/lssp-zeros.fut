-- Parallel Longest Satisfying Segment
--
-- ==
-- compiled input {
--    [1i32, -2, -2, 0, 0, 0, 0, 0, 3, 4, -6, 1]
-- }
-- output {
--    5
-- }
--
-- compiled input {
--    [1i32, 0, 0, 0, 0, 6, 7, 8, 0, 0]
--}
-- output {
--    4
--}
-- compiled input {
--    [1i32, 1, 1, 1, 1, 1, 1, 1, 1, 1]
--}
-- output {
--    0
--}
-- compiled input {
--    empty([0]i32)
--}
-- output {
--    0
--}
--
-- compiled input @ lssp-data.in
-- output @ data-zeros.out



import "lssp-seq"
import "lssp"

type int = i32

let main (xs: []int) : int =
  let pred1 x   = (x == 0)
  let pred2 x y = (x == 0) && (y == 0)
--  in  lssp_seq pred1 pred2 xs
  in  lssp pred1 pred2 xs

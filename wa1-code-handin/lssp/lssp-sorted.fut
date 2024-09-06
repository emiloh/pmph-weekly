-- Parallel Longest Satisfying Segment
--
-- ==
-- compiled input {
--    [1, -2, -2, 0, 0, 0, 0, 0, 3, 4, -6, 1]
-- }  
-- output { 
--    9
-- }
-- compiled input {
--   [1i32, 2, 3, 4, 5, -2, 7, 8] 
-- }
-- output {
--   5
-- }
-- compiled input {
--   [1i32, -1, 3, 4, 5, -2, 7, 8]
-- }
-- output {
--   4
-- }

import "lssp"
import "lssp-seq"

type int = i32

let main (xs: []int) : int =
  let pred1 _   = true
  let pred2 x y = (x <= y)
--  in  lssp_seq pred1 pred2 xs
  in  lssp pred1 pred2 xs

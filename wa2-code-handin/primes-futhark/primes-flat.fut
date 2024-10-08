-- Primes: Flat-Parallel Version
-- ==
-- compiled input { 30i64 } output { [2i64, 3i64, 5i64, 7i64, 11i64, 13i64, 17i64, 19i64, 23i64, 29i64] }
-- compiled input { 10000000i64 }

-- output @ ref10000000.out

let primesFlat (n : i64) : []i64 =
  let sq_primes   = [2i64, 3i64, 5i64, 7i64]
  let len  = 8i64
  let (sq_primes, _) =
    loop (sq_primes, len) while len < n do
      -- this is "len = min n (len*len)" 
      -- but without running out of i64 bounds 
      let len = if n / len < len then n else len*len

      let mult_lens = map (\ p -> (len / p) - 1 ) sq_primes
      let flat_size = reduce (+) 0 mult_lens

      --------------------------------------------------------------
      -- The current iteration knows the primes <= 'len', 
      --  based on which it will compute the primes <= 'len*len'
      -- ToDo: replace the dummy code below with the flat-parallel
      --       code that is equivalent with the nested-parallel one:
      --
      --   let composite = map (\ p -> let mm1 = (len / p) - 1
      --                               in  map (\ j -> j * p ) (map (+2) (iota mm1))
      --                       ) sq_primes
      --   let not_primes = reduce (++) [] composite
      --
      -- Your code should compute the correct `not_primes`.
      -- Please look at the lecture slides L2-Flattening.pdf to find
      --  the normalized nested-parallel version.
      -- Note that the scalar computation `mm1 = (len / p) - 1' has
      --  already been distributed and the result is stored in "mult_lens",
      --  where `p \in sq_primes`.
      -- Also note that `not_primes` has flat length equal to `flat_size`
      --  and the shape of `composite` is `mult_lens`. 

      ------------- Helpers -------------------------------

      let rotate [n] 'a (ne: a) (arr: [n] a) : [n] a = 
	map (\i -> if i == 0 then ne else arr[i-1]) (iota n)

      let scan_exc [n] 'a (op: a -> a -> a) (ne: a) (arr: [n] a) : [n] a = 
	scan op ne (rotate ne arr)

	
      let sgmScan [n] 't
		(op: t -> t -> t)
		(ne: t)
		(flags: [n]bool)
		(vals: [n]t)
		: [n]t =
	scan (\(f1, v1) (f2, v2) -> (f1 || f2, if f2 then v2 else op v1 v2))
	(false, ne)
	(zip flags vals)
	|> unzip
	|> (.1)

      let sgmScan_exc [n] 'a (op: a -> a -> a) (ne: a) (flags: [n] bool) (vals: [n] a) : [n] a =
	scan_exc (\(f1, v1) (f2, v2) -> (f1 || f2, if f2 then v2 else op v1 v2))
	(false, ne)
	(zip flags vals)
	|> unzip
	|> (.1)

      ------------- Helpers end ---------------------------
     
     let composite = 

	 -- Distribute iota 
	 let indicies = scan_exc (+) 0 mult_lens
	 let trues = map (\_ -> true) indicies
	 let flag = scatter (replicate flat_size false) indicies trues
	 let one_arr = replicate flat_size 1
	 let iot = sgmScan_exc (+) 0 flag one_arr

	 -- Distribute map 
	 let arr = map (+ 2) iot

	 -- Distribute replicate
	 let vals = scatter (replicate flat_size 0) indicies sq_primes
	 let ps = sgmScan (+) 0 flag vals

	 -- Distrubute map
	 let res = map2 (*) ps arr

	 in res

    
      let not_primes = composite

      -- If not_primes is correctly computed, then the remaining
      -- code is correct and will do the job of computing the prime
      -- numbers up to n!
      --------------------------------------------------------------
      --------------------------------------------------------------

       let zero_array = replicate flat_size 0i8
       let mostly_ones= map (\ x -> if x > 1 then 1i8 else 0i8) (iota (len+1))
       let prime_flags= scatter mostly_ones not_primes zero_array
       let sq_primes = filter (\i-> (i > 1i64) && (i <= n) && (prime_flags[i] > 0i8))
                              (0...len)

       in  (sq_primes, len)

  in sq_primes

-- RUN a big test with:
--   $ futhark cuda primes-flat.fut
--   $ echo "10000000i64" | ./primes-flat -t /dev/stderr -r 10 > /dev/null
-- or simply use futhark bench, i.e.,
--   $ futhark bench --backend=cuda primes-flat.fut
let main (n : i64) : []i64 = primesFlat n

(define (problem TPP)
(:domain TPP-Propositional)
(:objects
	Goods1 Goods2 Goods3 Goods4 - goods
	Truck1 - truck
	Market1 - market
	Depot1 - depot
	Level0 Level1 - level)

(:init
	(next Level1 Level0)
	(ready-to-load Goods1 Market1 Level0)
	(ready-to-load Goods2 Market1 Level0)
	(ready-to-load Goods3 Market1 Level0)
	(ready-to-load Goods4 Market1 Level0)
	(stored Goods1 Level0)
	(stored Goods2 Level0)
	(stored Goods3 Level0)
	(stored Goods4 Level0)
	(loaded Goods1 Truck1 Level0)
	(loaded Goods2 Truck1 Level0)
	(loaded Goods3 Truck1 Level0)
	(loaded Goods4 Truck1 Level0)
	(connected Depot1 Market1)
	(connected Market1 Depot1)
	(on-sale Goods1 Market1 Level1)
	(on-sale Goods2 Market1 Level1)
	(on-sale Goods3 Market1 Level1)
	(on-sale Goods4 Market1 Level1)
	(at Truck1 Depot1))

(:goal (and
	(stored Goods1 Level1)
	(stored Goods2 Level1)
	(stored Goods3 Level1)
	(stored Goods4 Level1)))

)
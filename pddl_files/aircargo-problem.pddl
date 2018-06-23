(define (problem Transport)

(:domain aircargo)

(:init 	(at C1 SFO)
	(at C2 JFK)
	(at P1 SFO)
	(at P2 JFK)
	(cargo C1)
	(cargo C2)
	(plane P1)
	(plane P2)
	(airport JFK)
	(airport SFO))

(:goal (and (at C1 JFK) (at C2 SFO)))
)

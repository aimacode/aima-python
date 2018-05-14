(define (problem SussmanAnomaly)

(:domain BlocksWorld)

(:init 	(clear C)
	(clear B)
	(on A Table)
	(on B Table)
	(on C A)
	(block A)
	(block B)
	(block C))

(:goal (and (on A B) (on B C)))
)

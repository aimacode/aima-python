(define (problem ThreeBlockTower)

   (:domain BlocksWorld)

   (:init
        (on A Table)
        (on B Table)
	    (on C Table)
	    (block A)
	    (block B)
	    (block C)
	    (clear A)
	    (clear B)
	    (clear C)
    )

    (:goal (and (on A B) (on B C)))

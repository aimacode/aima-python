(define (problem ThreeBlockTower)

(:domain BlocksWorld)

(:objects D B A C - block)

(:INIT 	(CLEAR C) 
	(CLEAR A) 
	(CLEAR B)
	(ONTABLE C) 
	(ONTABLE A)
	(ONTABLE B)
	(HANDEMPTY))

(:goal (AND (ON B C) (ON A B)))
)

(define (problem PutOnShoes)

   (:domain Shoes)

   (:init (clear LF)
          (clear RF)
          (leftfoot LF)
          (rightfoot RF)
   )

   (:goal (and (On RightShoe RF) (on LeftShoe LF)))
)

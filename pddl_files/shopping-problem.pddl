;; Going shopping

(define (problem GoingShopping)

   (:domain Shopping)

   (:init (At Home)
          (Loc Home)
          (Loc Supermarket)
          (Loc HardwareStore)
          (Sells Supermarket Milk)
          (Sells Supermarket Banana)
          (Sells HardwareStore Drill)
   )

   (:goal (and (Have Milk) (Have Banana) (Have Drill) (At Home)))
)

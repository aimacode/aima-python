;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Cake domain
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (domain Cake)
  (:requirements :strips)

  (:action Eat
	     :parameters (?x)
	     :precondition (have ?x)
	     :effect (and (eaten ?x) (not (have ?x)))
  )

  (:action Bake
         :parameters (?x)
         :precondition (not (have ?x))
         :effect (have ?x)
  )
)


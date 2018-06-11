;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Cake domain
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (domain Cake)
  (:requirements :strips)

  (:action Eat
	     :parameters (Cake)
	     :precondition (have Cake)
	     :effect (and (eaten Cake) (not (have Cake)))
  )

  (:action Bake
         :parameters (Cake)
         :precondition (not (have Cake))
         :effect (have Cake)
  )
)


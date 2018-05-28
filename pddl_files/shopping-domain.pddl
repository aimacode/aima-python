;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Shopping domain
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (domain Shopping)
  (:requirements :strips)

  (:action Buy
	     :parameters (?x ?store)
	     :precondition (and (at ?store) (sells ?store ?x))
	     :effect (have ?x)
  )

  (:action Go
         :parameters (?x ?y)
         :precondition (and (at ?x) (loc ?x) (loc ?y))
         :effect (and (at ?y) (not (at ?x)))
  )
)


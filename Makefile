unit:
    py.test -v
    
coverage:
    py.test --cov=tf_metabolism --cov-report term-missing --cov-report html
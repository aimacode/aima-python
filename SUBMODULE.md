This is a guide on how to update the `aima-data` submodule to the latest version. This needs to be done every time something changes in the [aima-data](https://github.com/aimacode/aima-data) repository. All the below commands should be executed from the local directory of the `aima-python` repository, using `git`.

```
git submodule deinit aima-data
git rm aima-data
git submodule add https://github.com/aimacode/aima-data.git aima-data
git commit
git push origin
```

Then you need to pull request the changes (unless you are a collaborator, in which case you can commit directly to the master).

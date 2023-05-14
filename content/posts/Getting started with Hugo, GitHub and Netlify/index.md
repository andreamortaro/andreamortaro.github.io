---
title: "Getting started with Hugo, GitHub and Netlify"
date: 2022-05-08
summary: "A quick guide to start blogging with these three tools."
tags: ["hugo", 'github','netlify','website']
---


## Why use Hugo?
In this post I want to explain how I setted up my static website using Hugo, GitHub and Netlify. I decided to use these tools because:
1. **Easy to use.** To write contents you will use Markdown;
2. **Keep focus on writing.** All your website contents will be built in Hugo;
3. **Fast build times.** By connecting Netlify with your GitHub repo you have a continuous deployment, which makes automatic the content publishing.

---

Let's get started. As a host operating system, I will be using **Mint 20.03**.

## Step 0: Install Hugo <cite> [^1]</cite>
[^1]: Follow the [guide](https://gohugo.io/getting-started/installing/) to install hugo in the proper way.

From the [official Hugo realeases page](https://github.com/gohugoio/hugo/releases) download a <kbd>.deb</kbd> package of the latest version and install it with <kbd>dpkg</kbd> command:
```
$ sudo dpkg -i hugo_<latest_version>_Linux-64bit.deb
```

## Step 1: Start with Hugo

### Create a new website
Use the <kbd>hugo new site</kbd> command:
```
$ hugo new site test-site
```
In this way the new site has a structure, but it has no content.

### Add a theme
From [official Hugo Themes](https://themes.gohugo.io/) choose a theme and then add the theme repository as a Git submodule as follows:
```
$ git submodule add -b stable https://github.com/jpanther/congo.git themes/congo
```

{{< alert >}}
**Warning!** In this tutorial we choose [Congo theme](https://github.com/jpanther/congo). Further configuration at [congo docs](https://jpanther.github.io/congo/docs/getting-started/).
{{< /alert >}}

Then modify <kbd>config.toml</kbd> by specifying the theme to use:

{{< highlight html "linenos=table,hl_lines=4" >}}
baseURL = "/"
languageCode = "en-us"
title = "My New Hugo Site"
theme = "congo"
{{< /highlight >}}

Pay attention to this step, otherwise there could be some problems.<cite> [^2]</cite>
[^2]: For more info take a look at the [offical docs](https://jpanther.github.io/congo/docs/installation/#set-up-theme-configuration-files).

To check our site we can start the built-in server to build and serve the site.<br>Run the <kbd>hugo server</kbd> command from the site’s root directory:
```
$ hugo server
```
and go to [http://localhost:1313/](http://localhost:1313/) in your browser.

### Write a post
Run the <kbd>hugo new</kbd> command:
```
$ hugo new post/new-post.md
```
In this way Hugo creates a new file <kbd>new-post.md</kbd> into <kbd>content/post</kbd>. It should looks like:
```
---
title: "New Post"
date: 2022-05-02T00:05:45+02:00
draft: true
---

```
as the <kbd>default.md</kbd> in <kbd>/archetypes</kbd> folder.<br> The new file has automatically setted the flag <kbd>draft: true</kbd>. With this flag you can control the visibility of the file.

Running locally <kbd>hugo server</kbd> we consider only the visible contents (i.e. the posts with <kbd>draft: false</kbd> flag), while <kbd>hugo server -D</kbd> allows us to consider all the contents (also the ones with <kbd>draft: true</kbd> flag).


### Generate static output for publishing

One way to publish our website is:
```
$ hugo -D
```
In this way Hugo generates your static HTML pages into the <kbd>./public</kbd>. Then we can copy the content of the folder to some server by yourself.

Anyway it's more convenient to use **Netlify** for this step.  Before to pass to Netlify we want to setup a new repository on Github.

## Step 2: Taking track with Git

### Create a local repo

To create a local repository:

```
$ git init
$ git add .
$ git commit -m 'start my website'
```

Then we have to setup a remote repository on GitHub.

### Sync your changes to a remote GitHub repository

Since it's not necessary to track also the generated HTML in <kbd>/public</kbd>, so we specify it in <kbd>.gitignore</kbd>.
~~~
$ git remote add origin https://github.com/<username>/<GitHub_repository_name>.git
$ echo 'public/' >> .gitignore
$ git branch -M main
$ git add .
$ git commit -m 'after the first post'
$ git push -u origin main
~~~

<!--(To make things more smooth: [SSH](:/21563ea4566c4f56a59dfd9647b374ad))-->
Now we pass to set up Netlify.

## Step 3: Setting up Netlify
First sign up to [Netlify](http://netlify.com/). Click on <kbd>New site from Git</kbd> button and then it will open a page titled "Create a new site".

The procedure is divided in three steps:
1. **"Connect to Git provider"**:\
Click on <kbd>GitHub</kbd> button to authorize Netlify to access your repositories, where your site's source code is hosted.
2. **"Pick a repository"**:\
Choose the repository where you store your website content (i.e the one you have created in the previous step.)

3. **"Build options, and deploy!"**.\
Some deployment configurations as:
    * <kbd>Owner</kbd> (the owner of the website)
    * <kbd>Branch deploy</kbd>  (which branch of the GitHub repo to deploy)
    * <kbd>build command</kbd>  (which command to use to build the site)
    * <kbd>publish directory</kbd> (the directory to deploy from, <kbd>public</kbd> is where Hugo generates the output by default).

Then click on the <kbd>Deploy site</kbd> and it will be presented the site overview.

In the <kbd>Change site name</kbd> pop-up window, set the new site name and save it.

## Troubleshooting
The first deploy may fail. Possible reasons are:
* **Specify to Netlify which Hugo version to use**:\
As in this answer on [stackoverflow](https://stackoverflow.com/questions/48045132/changing-hugo-version-on-netlify-build/48045133#48045133), if you do not setup the right version of Hugo, then the default one may be different and work differently as you see in local. A smooth solution is to specify the version you use in a config file <kbd>netlify.toml</kbd> at the site’s root directory.\
For [congo](https://github.com/jpanther/congo) theme I use this [<kbd>netlify.toml</kbd>](https://github.com/jpanther/congo/blob/dev/netlify.toml), which is a little bit different.

## Sources
To start my site and so for this tutorial I followed the steps described in:
* [dolugen.com](https://dolugen.com/how-to-publish-your-website-with-hugo-and-netlify/#generate-static-output-for-publishing)
* [kiroule.com](https://www.kiroule.com/article/start-blogging-with-github-hugo-and-netlify/)

## Conclusion
Now we have built our static website! Looking at the docs of the theme we can customize our site.

[dg]: https://dolugen.com/how-to-publish-your-website-with-hugo-and-netlify/#generate-static-output-for-publishing
[kir]: https://www.kiroule.com/article/start-blogging-with-github-hugo-and-netlify/


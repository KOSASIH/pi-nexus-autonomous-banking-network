// community/forum.js
class Forum {
    constructor() {
        this.posts = []; // Store forum posts
    }

    createPost(title, content, author) {
        const post = {
            id: this.posts.length + 1,
            title: title,
            content: content,
            author: author,
            comments: [],
            createdAt: new Date(),
        };
        this.posts.push(post);
        console.log(`Post created: ${title}`);
        return post;
    }

    getPosts() {
        return this.posts;
    }

    getPostById(postId) {
        const post = this.posts.find(p => p.id === postId);
        if (!post) {
            throw new Error('Post not found.');
        }
        return post;
    }

    addComment(postId, commentContent, author) {
        const post = this.getPostById(postId);
        const comment = {
            id: post.comments.length + 1,
            content: commentContent,
            author: author,
            createdAt: new Date(),
        };
        post.comments.push(comment);
        console.log(`Comment added to post ${postId}: ${commentContent}`);
        return comment;
    }
}

module.exports = Forum;
